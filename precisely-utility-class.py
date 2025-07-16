import asyncio
import aiohttp
import aioredis
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import TransportQueryError
import hashlib
from functools import lru_cache
import backoff

logger = logging.getLogger(__name__)

@dataclass
class BuildingFootprint:
    building_id: str
    geometry: Dict
    building_type: str
    area: float
    elevation: Optional[float] = None
    height: Optional[float] = None
    attributes: Dict = None

@dataclass
class ParcelFootprint:
    parcel_id: str
    geometry: Dict
    area: float
    apn: Optional[str] = None
    fips: Optional[str] = None
    attributes: Dict = None

@dataclass
class QueryMetadata:
    count: int
    total_pages: int
    current_page: int
    credits_used: float

class TokenManager:
    """Manages token generation and refresh"""
    
    def __init__(self, client_id: str, client_secret: str, auth_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self._token = None
        self._token_expiry = None
        self._lock = asyncio.Lock()
    
    async def get_token(self, session: aiohttp.ClientSession) -> str:
        """Get valid token, refreshing if necessary"""
        async with self._lock:
            if self._is_token_valid():
                return self._token
            
            return await self._refresh_token(session)
    
    def _is_token_valid(self) -> bool:
        """Check if current token is still valid"""
        if not self._token or not self._token_expiry:
            return False
        
        # Refresh 5 minutes before expiry
        buffer_time = timedelta(minutes=5)
        return datetime.utcnow() < (self._token_expiry - buffer_time)
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _refresh_token(self, session: aiohttp.ClientSession) -> str:
        """Refresh authentication token"""
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        async with session.post(self.auth_url, data=auth_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Token generation failed: {error_text}")
            
            token_data = await response.json()
            self._token = token_data['access_token']
            expires_in = token_data.get('expires_in', 3600)
            self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
            
            logger.info(f"Token refreshed, expires at {self._token_expiry}")
            return self._token

class CreditTracker:
    """Tracks credit usage across queries"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.local_cache = {}
        self._lock = asyncio.Lock()
    
    async def track_usage(self, credits: float, query_type: str):
        """Track credit usage"""
        async with self._lock:
            # Update local cache
            today = datetime.utcnow().strftime('%Y-%m-%d')
            key = f"credits:{today}"
            
            # Update Redis
            await self.redis.hincrbyfloat(key, 'total', credits)
            await self.redis.hincrbyfloat(key, query_type, credits)
            await self.redis.expire(key, 86400 * 30)  # Keep for 30 days
            
            # Update local cache
            if today not in self.local_cache:
                self.local_cache[today] = {'total': 0}
            
            self.local_cache[today]['total'] += credits
            self.local_cache[today][query_type] = self.local_cache[today].get(query_type, 0) + credits
    
    async def get_usage_summary(self, days: int = 7) -> Dict:
        """Get credit usage summary"""
        summary = {}
        
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
            key = f"credits:{date}"
            
            usage = await self.redis.hgetall(key)
            if usage:
                summary[date] = {k.decode(): float(v) for k, v in usage.items()}
        
        return summary

class PreciselyUtility:
    """Scalable GraphQL client for Precisely Data Graph API"""
    
    def __init__(self, client_id: str, client_secret: str, 
                 redis_url: str = "redis://localhost:6379",
                 cache_ttl: int = 3600,
                 max_retries: int = 3):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://api.cloud.precisely.com"
        self.graphql_endpoint = f"{self.base_url}/data-graph/graphql"
        self.auth_url = f"{self.base_url}/oauth/token"
        
        # Configuration
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        # Will be initialized in async context
        self.client = None
        self.session = None
        self.redis = None
        self.token_manager = TokenManager(client_id, client_secret, self.auth_url)
        self.credit_tracker = None
        
        # Query templates
        self._init_query_templates()
    
    def _init_query_templates(self):
        """Initialize GraphQL query templates"""
        self.queries = {
            'building_by_address': """
                query GetBuildingByAddress($address: String!) {
                    getByAddress(address: $address) {
                        queryCredits
                        buildings {
                            metadata {
                                count
                                pageNumber
                                totalPages
                            }
                            data {
                                buildingID
                                buildingType {
                                    value
                                    description
                                }
                                geometry
                                buildingArea
                                elevation
                                maximumElevation
                                minimumElevation
                                ubid
                                fips
                                geographyID
                                latitude
                                longitude
                            }
                        }
                    }
                }
            """,
            
            'parcel_by_address': """
                query GetParcelByAddress($address: String!) {
                    getByAddress(address: $address) {
                        queryCredits
                        parcels {
                            metadata {
                                count
                                pageNumber
                                totalPages
                            }
                            data {
                                parcelID
                                geometry
                                parcelArea
                                apn
                                fips
                                geographyID
                                latitude
                                longitude
                                elevation
                            }
                        }
                    }
                }
            """,
            
            'building_and_parcel': """
                query GetBuildingAndParcel($address: String!) {
                    getByAddress(address: $address) {
                        queryCredits
                        buildings {
                            metadata {
                                count
                            }
                            data {
                                buildingID
                                geometry
                                buildingArea
                                buildingType {
                                    value
                                }
                            }
                        }
                        parcels {
                            metadata {
                                count
                            }
                            data {
                                parcelID
                                geometry
                                parcelArea
                                apn
                            }
                        }
                    }
                }
            """,
            
            'metadata_only': """
                query GetMetadata($address: String!) {
                    getByAddress(address: $address) {
                        queryCredits
                        buildings {
                            metadata {
                                count
                                totalPages
                            }
                        }
                        parcels {
                            metadata {
                                count
                                totalPages
                            }
                        }
                    }
                }
            """,
            
            'building_by_id': """
                query GetBuildingById($id: String!) {
                    getById(id: $id, queryType: BUILDING_ID) {
                        queryCredits
                        buildings {
                            data {
                                buildingID
                                geometry
                                buildingArea
                                buildingType {
                                    value
                                }
                            }
                        }
                    }
                }
            """,
            
            'parcel_by_id': """
                query GetParcelById($id: String!) {
                    getById(id: $id, queryType: PARCEL_ID) {
                        queryCredits
                        parcels {
                            data {
                                parcelID
                                geometry
                                parcelArea
                                apn
                            }
                        }
                    }
                }
            """
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize async resources"""
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
        )
        
        # Initialize Redis
        self.redis = await aioredis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=False
        )
        
        # Initialize credit tracker
        self.credit_tracker = CreditTracker(self.redis)
        
        # Initialize GraphQL client
        await self._init_graphql_client()
    
    async def _init_graphql_client(self):
        """Initialize GraphQL client with authentication"""
        token = await self.token_manager.get_token(self.session)
        
        transport = AIOHTTPTransport(
            url=self.graphql_endpoint,
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
        )
        
        self.client = Client(
            transport=transport,
            fetch_schema_from_transport=False,
            execute_timeout=30
        )
    
    async def close(self):
        """Close all connections"""
        if self.session:
            await self.session.close()
        if self.redis:
            await self.redis.close()
        if self.client:
            await self.client.close_async()
    
    def _get_cache_key(self, query_type: str, params: Dict) -> str:
        """Generate cache key for query"""
        key_data = f"{query_type}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache"""
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        return None
    
    async def _set_cache(self, cache_key: str, data: Dict):
        """Set data in cache"""
        try:
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _execute_query(self, query_name: str, variables: Dict) -> Dict:
        """Execute GraphQL query with retry logic"""
        # Refresh token if needed
        token = await self.token_manager.get_token(self.session)
        
        # Update transport headers
        self.client.transport.headers['Authorization'] = f'Bearer {token}'
        
        # Execute query
        query = gql(self.queries[query_name])
        
        try:
            async with self.client as session:
                result = await session.execute(query, variable_values=variables)
            return result
        except TransportQueryError as e:
            logger.error(f"GraphQL query error: {e}")
            raise
    
    async def get_building_footprint(self, address: str, use_cache: bool = True) -> Tuple[Optional[BuildingFootprint], float]:
        """
        Get building footprint for an address
        
        Returns:
            Tuple of (BuildingFootprint, credits_used)
        """
        # Check cache
        cache_key = self._get_cache_key('building', {'address': address})
        
        if use_cache:
            cached = await self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Building footprint cache hit for: {address}")
                return BuildingFootprint(**cached['data']), 0.0
        
        # Execute query
        try:
            result = await self._execute_query('building_by_address', {'address': address})
            
            # Extract data
            credits_used = result['getByAddress']['queryCredits']
            buildings_data = result['getByAddress']['buildings']['data']
            
            if not buildings_data:
                return None, credits_used
            
            # Get first building (primary)
            building = buildings_data[0]
            
            footprint = BuildingFootprint(
                building_id=building['buildingID'],
                geometry=building['geometry'],
                building_type=building['buildingType']['value'],
                area=building.get('buildingArea', 0),
                elevation=building.get('elevation'),
                attributes={
                    'ubid': building.get('ubid'),
                    'fips': building.get('fips'),
                    'lat': building.get('latitude'),
                    'lon': building.get('longitude')
                }
            )
            
            # Cache result
            await self._set_cache(cache_key, {
                'data': asdict(footprint),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Track credits
            await self.credit_tracker.track_usage(credits_used, 'building')
            
            return footprint, credits_used
            
        except Exception as e:
            logger.error(f"Error getting building footprint: {e}")
            raise
    
    async def get_parcel_footprint(self, address: str, use_cache: bool = True) -> Tuple[Optional[ParcelFootprint], float]:
        """
        Get parcel footprint for an address
        
        Returns:
            Tuple of (ParcelFootprint, credits_used)
        """
        # Check cache
        cache_key = self._get_cache_key('parcel', {'address': address})
        
        if use_cache:
            cached = await self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Parcel footprint cache hit for: {address}")
                return ParcelFootprint(**cached['data']), 0.0
        
        # Execute query
        try:
            result = await self._execute_query('parcel_by_address', {'address': address})
            
            # Extract data
            credits_used = result['getByAddress']['queryCredits']
            parcels_data = result['getByAddress']['parcels']['data']
            
            if not parcels_data:
                return None, credits_used
            
            # Get first parcel (primary)
            parcel = parcels_data[0]
            
            footprint = ParcelFootprint(
                parcel_id=parcel['parcelID'],
                geometry=parcel['geometry'],
                area=parcel.get('parcelArea', 0),
                apn=parcel.get('apn'),
                fips=parcel.get('fips'),
                attributes={
                    'lat': parcel.get('latitude'),
                    'lon': parcel.get('longitude'),
                    'elevation': parcel.get('elevation')
                }
            )
            
            # Cache result
            await self._set_cache(cache_key, {
                'data': asdict(footprint),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Track credits
            await self.credit_tracker.track_usage(credits_used, 'parcel')
            
            return footprint, credits_used
            
        except Exception as e:
            logger.error(f"Error getting parcel footprint: {e}")
            raise
    
    async def get_both_footprints(self, address: str, use_cache: bool = True) -> Tuple[Optional[BuildingFootprint], Optional[ParcelFootprint], float]:
        """
        Get both building and parcel footprints in one query (more efficient)
        
        Returns:
            Tuple of (BuildingFootprint, ParcelFootprint, credits_used)
        """
        # Check cache
        cache_key = self._get_cache_key('both', {'address': address})
        
        if use_cache:
            cached = await self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Both footprints cache hit for: {address}")
                building = BuildingFootprint(**cached['building']) if cached.get('building') else None
                parcel = ParcelFootprint(**cached['parcel']) if cached.get('parcel') else None
                return building, parcel, 0.0
        
        # Execute query
        try:
            result = await self._execute_query('building_and_parcel', {'address': address})
            
            # Extract data
            credits_used = result['getByAddress']['queryCredits']
            buildings_data = result['getByAddress']['buildings']['data']
            parcels_data = result['getByAddress']['parcels']['data']
            
            # Process building
            building_footprint = None
            if buildings_data:
                building = buildings_data[0]
                building_footprint = BuildingFootprint(
                    building_id=building['buildingID'],
                    geometry=building['geometry'],
                    building_type=building['buildingType']['value'],
                    area=building.get('buildingArea', 0)
                )
            
            # Process parcel
            parcel_footprint = None
            if parcels_data:
                parcel = parcels_data[0]
                parcel_footprint = ParcelFootprint(
                    parcel_id=parcel['parcelID'],
                    geometry=parcel['geometry'],
                    area=parcel.get('parcelArea', 0),
                    apn=parcel.get('apn')
                )
            
            # Cache result
            cache_data = {
                'building': asdict(building_footprint) if building_footprint else None,
                'parcel': asdict(parcel_footprint) if parcel_footprint else None,
                'timestamp': datetime.utcnow().isoformat()
            }
            await self._set_cache(cache_key, cache_data)
            
            # Track credits
            await self.credit_tracker.track_usage(credits_used, 'both')
            
            return building_footprint, parcel_footprint, credits_used
            
        except Exception as e:
            logger.error(f"Error getting both footprints: {e}")
            raise
    
    async def get_metadata(self, address: str) -> Tuple[Dict, float]:
        """
        Get metadata only (count, pages) without full data
        Useful for checking availability before full query
        
        Returns:
            Tuple of (metadata_dict, credits_used)
        """
        try:
            result = await self._execute_query('metadata_only', {'address': address})
            
            credits_used = result['getByAddress']['queryCredits']
            
            metadata = {
                'buildings': {
                    'count': result['getByAddress']['buildings']['metadata']['count'],
                    'total_pages': result['getByAddress']['buildings']['metadata']['totalPages']
                },
                'parcels': {
                    'count': result['getByAddress']['parcels']['metadata']['count'],
                    'total_pages': result['getByAddress']['parcels']['metadata']['totalPages']
                }
            }
            
            # Track credits (metadata queries are usually cheaper)
            await self.credit_tracker.track_usage(credits_used, 'metadata')
            
            return metadata, credits_used
            
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            raise
    
    async def check_credits(self, days: int = 7) -> Dict:
        """
        Check credit usage for the past N days
        
        Returns:
            Dict with daily and total credit usage
        """
        usage = await self.credit_tracker.get_usage_summary(days)
        
        # Calculate totals
        total_credits = 0
        total_by_type = {}
        
        for day, day_usage in usage.items():
            total_credits += day_usage.get('total', 0)
            
            for query_type, credits in day_usage.items():
                if query_type != 'total':
                    total_by_type[query_type] = total_by_type.get(query_type, 0) + credits
        
        return {
            'daily_usage': usage,
            'total_credits': total_credits,
            'total_by_type': total_by_type,
            'average_daily': total_credits / days if days > 0 else 0
        }
    
    async def get_building_by_id(self, building_id: str) -> Tuple[Optional[BuildingFootprint], float]:
        """Get building by PreciselyID"""
        try:
            result = await self._execute_query('building_by_id', {'id': building_id})
            
            credits_used = result['getById']['queryCredits']
            buildings_data = result['getById']['buildings']['data']
            
            if not buildings_data:
                return None, credits_used
            
            building = buildings_data[0]
            footprint = BuildingFootprint(
                building_id=building['buildingID'],
                geometry=building['geometry'],
                building_type=building['buildingType']['value'],
                area=building.get('buildingArea', 0)
            )
            
            await self.credit_tracker.track_usage(credits_used, 'building_by_id')
            
            return footprint, credits_used
            
        except Exception as e:
            logger.error(f"Error getting building by ID: {e}")
            raise
    
    async def batch_get_footprints(self, addresses: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Batch process multiple addresses
        
        Returns:
            List of results with address, building, parcel, and credits
        """
        results = []
        
        # Process in batches
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.get_both_footprints(address)
                for address in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Format results
            for address, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append({
                        'address': address,
                        'error': str(result),
                        'building': None,
                        'parcel': None,
                        'credits': 0
                    })
                else:
                    building, parcel, credits = result
                    results.append({
                        'address': address,
                        'building': asdict(building) if building else None,
                        'parcel': asdict(parcel) if parcel else None,
                        'credits': credits
                    })
        
        return results
    
    async def clear_cache(self, pattern: str = "*"):
        """Clear cache entries matching pattern"""
        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis.delete(*keys)
                if cursor == 0:
                    break
            logger.info(f"Cache cleared for pattern: {pattern}")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Example usage
async def main():
    # Initialize utility
    async with PreciselyUtility(
        client_id="your_client_id",
        client_secret="your_client_secret"
    ) as utility:
        
        # Example 1: Get building footprint
        building, credits = await utility.get_building_footprint("3252 Fulton St Denver")
        print(f"Building: {building}")
        print(f"Credits used: {credits}")
        
        # Example 2: Get metadata first to check availability
        metadata, meta_credits = await utility.get_metadata("3252 Fulton St Denver")
        print(f"Available data: {metadata}")
        
        # Example 3: Get both footprints efficiently
        building, parcel, credits = await utility.get_both_footprints("3252 Fulton St Denver")
        print(f"Building: {building}")
        print(f"Parcel: {parcel}")
        
        # Example 4: Batch processing
        addresses = [
            "3252 Fulton St Denver",
            "10425 E 31st Ave Denver CO",
            "2903 Havana St Denver"
        ]
        results = await utility.batch_get_footprints(addresses)
        
        # Example 5: Check credit usage
        usage = await utility.check_credits(days=7)
        print(f"Credit usage summary: {usage}")


if __name__ == "__main__":
    asyncio.run(main())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````






import base64
import requests
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport


class PreciselyUtility:
    def __init__(self, client_id: str, client_secret: str, enable_cache: bool = False, credit_threshold: int = 500):
        self.client_id = client_id
        self.client_secret = client_secret
        self.enable_cache = enable_cache
        self.token = None
        self.cache = {}
        self.total_credits_used = 0
        self.credit_threshold = credit_threshold

    # --------------------- AUTH ---------------------
    def _get_token(self):
        if self.token is not None:
            return self.token

        creds = f"{self.client_id}:{self.client_secret}"
        basic = base64.b64encode(creds.encode()).decode()
        headers = {
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}

        response = requests.post("https://api.precisely.com/oauth/token", headers=headers, data=data)
        response.raise_for_status()
        self.token = response.json()["access_token"]
        return self.token

    def _get_graphql_client(self):
        transport = RequestsHTTPTransport(
            url="https://api.cloud.precisely.com/data-graph/graphql",
            headers={"Authorization": f"Bearer {self._get_token()}"},
            verify=True,
            retries=2
        )
        return Client(transport=transport, fetch_schema_from_transport=True)

    def _track_credits(self, credits: int):
        self.total_credits_used += credits
        if self.total_credits_used > self.credit_threshold:
            print(f"[⚠️ Warning] Total estimated usage: {self.total_credits_used} credits exceeds threshold.")

    # --------------------- METADATA ---------------------
    def get_query_metadata_and_credits(self, address: str) -> dict:
        client = self._get_graphql_client()
        query = gql("""
        query($addr: String!) {
          getByAddress(address: $addr) {
            queryCredits
            addresses {
              metadata { pageNumber totalPages count }
            }
            buildings {
              metadata { pageNumber totalPages count }
            }
            parcels {
              metadata { pageNumber totalPages count }
            }
          }
        }
        """)
        try:
            result = client.execute(query, variable_values={"addr": address})
            return result.get("getByAddress", {})
        except Exception as e:
            print(f"[Metadata Error] {e}")
            return {"error": str(e)}

    def get_spatial_metadata_and_credits(self, lat: float, lon: float, dataset: str = "buildings") -> dict:
        if dataset not in {"buildings", "parcels"}:
            raise ValueError("dataset must be 'buildings' or 'parcels'")

        query = gql(f"""
        query($lat: Float!, $lon: Float!) {{
          getBySpatial(
            spatialFunction: INTERSECTS,
            geoJson: {{
              type: "Point",
              coordinates: [{{ lon: $lon, lat: $lat }}]
            }}
          ) {{
            queryCredits
            {dataset} {{
              metadata {{
                count
                pageNumber
                totalPages
              }}
            }}
          }}
        }}
        """)

        client = self._get_graphql_client()
        try:
            return client.execute(query, variable_values={"lat": lat, "lon": lon}).get("getBySpatial", {})
        except Exception as e:
            print(f"[Spatial Metadata Error] {e}")
            return {"error": str(e)}

    # --------------------- FORWARD GEOCODE ---------------------
    def get_footprint_wkt(self, address: str, include_parcel=True, include_building=True, max_credits=5):
        if self.enable_cache and address in self.cache:
            return self.cache[address]

        meta = self.get_query_metadata_and_credits(address)
        cost = meta.get("queryCredits", 999)
        self._track_credits(cost)

        if cost > max_credits:
            print(f"[❌ Skipped] Address '{address}' would cost {cost} credits.")
            return {"skipped": True, "cost": cost}

        fields = []
        if include_building:
            fields.append("""
            buildings {
              data {
                geometry
              }
            }
            """)
        if include_parcel:
            fields.append("""
            parcels {
              data {
                geometry
              }
            }
            """)

        if not fields:
            return {"error": "Must include at least one of parcel or building."}

        query_template = f"""
        query($addr: String!) {{
          getByAddress(address: $addr) {{
            {' '.join(fields)}
          }}
        }}
        """
        query = gql(query_template)
        client = self._get_graphql_client()

        try:
            result = client.execute(query, variable_values={"addr": address})
            out = {"building": [], "parcel": []}

            if include_building:
                buildings = result.get("getByAddress", {}).get("buildings", {}).get("data", [])
                out["building"] = [b["geometry"] for b in buildings if b.get("geometry")]

            if include_parcel:
                parcels = result.get("getByAddress", {}).get("parcels", {}).get("data", [])
                out["parcel"] = [p["geometry"] for p in parcels if p.get("geometry")]

            if self.enable_cache:
                self.cache[address] = out

            return out
        except Exception as e:
            print(f"[Precisely Error] {e}")
            return {"error": str(e)}

    # --------------------- REVERSE GEOCODE ---------------------
    def reverse_geocode_from_latlon(self, lat: float, lon: float, prefer='building', max_credits=5):
        if prefer not in {"building", "parcel"}:
            raise ValueError("prefer must be 'building' or 'parcel'")

        dataset = f"{prefer}s"

        meta = self.get_spatial_metadata_and_credits(lat, lon, dataset)
        cost = meta.get("queryCredits", 999)
        self._track_credits(cost)

        if cost > max_credits:
            print(f"[❌ Skipped Reverse Geocode] Cost {cost} exceeds threshold.")
            return None

        query = gql(f"""
        query($lat: Float!, $lon: Float!) {{
          getBySpatial(
            spatialFunction: INTERSECTS,
            geoJson: {{
              type: "Point",
              coordinates: [{{ lon: $lon, lat: $lat }}]
            }}
          ) {{
            {dataset} {{
              data {{
                geometry
              }}
            }}
          }}
        }}
        """)
        client = self._get_graphql_client()

        try:
            result = client.execute(query, variable_values={"lat": lat, "lon": lon})
            data = result.get("getBySpatial", {}).get(dataset, {}).get("data", [])
            return data[0].get("geometry") if data else None
        except Exception as e:
            print(f"[Reverse Geocode Error] {e}")
            return None

