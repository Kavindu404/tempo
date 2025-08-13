import requests
from typing import Tuple, Optional, Dict, Any
import urllib.parse


class GoogleUtilityError(Exception):
    """Custom exception for GoogleUtility errors"""
    pass


class GoogleUtility:
    """
    A utility class for Google Geocoding API operations.
    
    Note: Google Geocoding API uses simple API key authentication, 
    not API key + secret with token generation.
    """
    
    BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def __init__(self, api_key: str, secret: Optional[str] = None):
        """
        Initialize the GoogleUtility class.
        
        Args:
            api_key (str): Your Google Maps API key
            secret (Optional[str]): Not used by Google Geocoding API, 
                                  included for interface compatibility
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.secret = secret  # Not used by Google Geocoding API
        self.session = requests.Session()
        
        # Google Geocoding API doesn't require token generation
        # but we'll call _genToken() to maintain the requested interface
        self.token = self._genToken()
    
    def _genToken(self) -> Optional[str]:
        """
        Generate token if needed.
        
        Note: Google Geocoding API doesn't use tokens, but this method
        is included to match the requested interface.
        
        Returns:
            Optional[str]: None (Google Geocoding API doesn't use tokens)
        """
        # Google Geocoding API uses simple API key authentication
        # No token generation needed
        return None
    
    def geocode(self, address: str) -> Tuple[float, float]:
        """
        Convert an address to latitude and longitude coordinates.
        
        Args:
            address (str): The address to geocode
            
        Returns:
            Tuple[float, float]: A tuple containing (latitude, longitude)
            
        Raises:
            GoogleUtilityError: If geocoding fails or returns no results
            ValueError: If address is empty or invalid
        """
        if not address or not address.strip():
            raise ValueError("Address cannot be empty")
        
        # URL encode the address
        encoded_address = urllib.parse.quote_plus(address.strip())
        
        # Build the request URL
        url = f"{self.BASE_URL}?address={encoded_address}&key={self.api_key}"
        
        try:
            # Make the API request
            response = self.session.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse JSON response
            data = response.json()
            
            # Check API response status
            status = data.get('status')
            
            if status == 'OK':
                # Extract coordinates from the first result
                if data.get('results') and len(data['results']) > 0:
                    location = data['results'][0]['geometry']['location']
                    lat = location['lat']
                    lng = location['lng']
                    return (lat, lng)
                else:
                    raise GoogleUtilityError("No results found for the given address")
            
            elif status == 'ZERO_RESULTS':
                raise GoogleUtilityError(f"No results found for address: {address}")
            
            elif status == 'OVER_QUERY_LIMIT':
                raise GoogleUtilityError("API query limit exceeded")
            
            elif status == 'REQUEST_DENIED':
                raise GoogleUtilityError("API request denied. Check your API key and permissions")
            
            elif status == 'INVALID_REQUEST':
                raise GoogleUtilityError("Invalid request. Check the address format")
            
            elif status == 'UNKNOWN_ERROR':
                raise GoogleUtilityError("Unknown error occurred on Google's servers")
            
            else:
                raise GoogleUtilityError(f"Geocoding failed with status: {status}")
                
        except requests.exceptions.RequestException as e:
            raise GoogleUtilityError(f"Network error occurred: {str(e)}")
        
        except ValueError as e:
            # JSON parsing error
            raise GoogleUtilityError(f"Invalid response format from API: {str(e)}")
        
        except Exception as e:
            raise GoogleUtilityError(f"Unexpected error occurred: {str(e)}")
    
    def reverse_geocode(self, lat: float, lng: float) -> str:
        """
        Convert latitude and longitude coordinates to an address.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            
        Returns:
            str: The formatted address
            
        Raises:
            GoogleUtilityError: If reverse geocoding fails
            ValueError: If coordinates are invalid
        """
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            raise ValueError("Latitude and longitude must be numbers")
        
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        
        if not (-180 <= lng <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        
        # Build the request URL for reverse geocoding
        url = f"{self.BASE_URL}?latlng={lat},{lng}&key={self.api_key}"
        
        try:
            # Make the API request
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check API response status
            status = data.get('status')
            
            if status == 'OK':
                # Return the formatted address from the first result
                if data.get('results') and len(data['results']) > 0:
                    return data['results'][0]['formatted_address']
                else:
                    raise GoogleUtilityError("No results found for the given coordinates")
            
            elif status == 'ZERO_RESULTS':
                raise GoogleUtilityError(f"No address found for coordinates: ({lat}, {lng})")
            
            else:
                raise GoogleUtilityError(f"Reverse geocoding failed with status: {status}")
                
        except requests.exceptions.RequestException as e:
            raise GoogleUtilityError(f"Network error occurred: {str(e)}")
        
        except ValueError as e:
            raise GoogleUtilityError(f"Invalid response format from API: {str(e)}")
        
        except Exception as e:
            raise GoogleUtilityError(f"Unexpected error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize the utility (replace with your actual API key)
    try:
        geocoder = GoogleUtility(api_key="YOUR_API_KEY_HERE")
        
        # Geocode an address
        address = "1600 Amphitheatre Parkway, Mountain View, CA"
        lat, lng = geocoder.geocode(address)
        print(f"Address: {address}")
        print(f"Coordinates: ({lat}, {lng})")
        
        # Reverse geocode coordinates
        reverse_address = geocoder.reverse_geocode(lat, lng)
        print(f"Reverse geocoded address: {reverse_address}")
        
    except GoogleUtilityError as e:
        print(f"GoogleUtility error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
































import requests
from typing import Tuple, Optional, Dict, Any
import urllib.parse


class GoogleUtilityError(Exception):
    """Custom exception for GoogleUtility errors"""
    pass


class GoogleUtility:
    """
    A utility class for Google Geocoding API operations.
    
    Note: Google Geocoding API uses simple API key authentication, 
    not API key + secret with token generation.
    """
    
    BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def __init__(self, api_key: str, secret: Optional[str] = None):
        """
        Initialize the GoogleUtility class.
        
        Args:
            api_key (str): Your Google Maps API key
            secret (Optional[str]): Not used by Google Geocoding API, 
                                  included for interface compatibility
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.secret = secret  # Not used by Google Geocoding API
        self.session = requests.Session()
        
        # Google Geocoding API doesn't require token generation
        # but we'll call _genToken() to maintain the requested interface
        self.token = self._genToken()
    
    def _genToken(self) -> Optional[str]:
        """
        Generate token if needed.
        
        Note: Google Geocoding API doesn't use tokens, but this method
        is included to match the requested interface.
        
        Returns:
            Optional[str]: None (Google Geocoding API doesn't use tokens)
        """
        # Google Geocoding API uses simple API key authentication
        # No token generation needed
        return None
    
    def geocode(self, address: str) -> Tuple[float, float]:
        """
        Convert an address to latitude and longitude coordinates.
        
        Args:
            address (str): The address to geocode
            
        Returns:
            Tuple[float, float]: A tuple containing (latitude, longitude)
            
        Raises:
            GoogleUtilityError: If geocoding fails or returns no results
            ValueError: If address is empty or invalid
        """
        if not address or not address.strip():
            raise ValueError("Address cannot be empty")
        
        # URL encode the address
        encoded_address = urllib.parse.quote_plus(address.strip())
        
        # Build the request URL
        url = f"{self.BASE_URL}?address={encoded_address}&key={self.api_key}"
        
        try:
            # Make the API request
            response = self.session.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse JSON response
            data = response.json()
            
            # Check API response status
            status = data.get('status')
            
            if status == 'OK':
                # Extract coordinates from the first result
                if data.get('results') and len(data['results']) > 0:
                    location = data['results'][0]['geometry']['location']
                    lat = location['lat']
                    lng = location['lng']
                    return (lat, lng)
                else:
                    raise GoogleUtilityError("No results found for the given address")
            
            elif status == 'ZERO_RESULTS':
                raise GoogleUtilityError(f"No results found for address: {address}")
            
            elif status == 'OVER_QUERY_LIMIT':
                raise GoogleUtilityError("API query limit exceeded")
            
            elif status == 'REQUEST_DENIED':
                raise GoogleUtilityError("API request denied. Check your API key and permissions")
            
            elif status == 'INVALID_REQUEST':
                raise GoogleUtilityError("Invalid request. Check the address format")
            
            elif status == 'UNKNOWN_ERROR':
                raise GoogleUtilityError("Unknown error occurred on Google's servers")
            
            else:
                raise GoogleUtilityError(f"Geocoding failed with status: {status}")
                
        except requests.exceptions.RequestException as e:
            raise GoogleUtilityError(f"Network error occurred: {str(e)}")
        
        except ValueError as e:
            # JSON parsing error
            raise GoogleUtilityError(f"Invalid response format from API: {str(e)}")
        
        except Exception as e:
            raise GoogleUtilityError(f"Unexpected error occurred: {str(e)}")
    
    def reverse_geocode(self, lat: float, lng: float) -> str:
        """
        Convert latitude and longitude coordinates to an address.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            
        Returns:
            str: The formatted address
            
        Raises:
            GoogleUtilityError: If reverse geocoding fails
            ValueError: If coordinates are invalid
        """
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            raise ValueError("Latitude and longitude must be numbers")
        
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        
        if not (-180 <= lng <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        
        # Build the request URL for reverse geocoding
        url = f"{self.BASE_URL}?latlng={lat},{lng}&key={self.api_key}"
        
        try:
            # Make the API request
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check API response status
            status = data.get('status')
            
            if status == 'OK':
                # Return the formatted address from the first result
                if data.get('results') and len(data['results']) > 0:
                    return data['results'][0]['formatted_address']
                else:
                    raise GoogleUtilityError("No results found for the given coordinates")
            
            elif status == 'ZERO_RESULTS':
                raise GoogleUtilityError(f"No address found for coordinates: ({lat}, {lng})")
            
            else:
                raise GoogleUtilityError(f"Reverse geocoding failed with status: {status}")
                
        except requests.exceptions.RequestException as e:
            raise GoogleUtilityError(f"Network error occurred: {str(e)}")
        
        except ValueError as e:
            raise GoogleUtilityError(f"Invalid response format from API: {str(e)}")
        
        except Exception as e:
            raise GoogleUtilityError(f"Unexpected error occurred: {str(e)}")


# CSV Processing with Parallel Geocoding
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    from pathlib import Path
    import logging
    import threading
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('geocoding.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    API_KEY = "YOUR_GOOGLE_API_KEY_HERE"  # Replace with your actual API key
    INPUT_FILE = "input_addresses.csv"    # Replace with your input file path
    OUTPUT_FILE = "output_addresses.csv"  # Replace with your desired output file path
    MAX_WORKERS = 5  # Adjust based on your API quota and needs
    DELAY_BETWEEN_REQUESTS = 0.2  # Adjust to respect rate limits
    
    # Statistics tracking
    stats = {
        'total_rows': 0,
        'rows_with_coords': 0,
        'rows_to_geocode': 0,
        'successful_geocodes': 0,
        'failed_geocodes': 0,
        'start_time': None,
        'end_time': None
    }
    
    # Thread lock for statistics
    lock = threading.Lock()
    
    def needs_geocoding(row):
        """Check if a row needs geocoding (has missing lat/lon coordinates)."""
        lat = row.get('latitude_precisely')
        lon = row.get('longitude_precisely')
        
        # Check if lat/lon are null, NaN, empty string, or zero
        lat_missing = pd.isna(lat) or lat == '' or lat == 0
        lon_missing = pd.isna(lon) or lon == '' or lon == 0
        
        return lat_missing or lon_missing
    
    def geocode_single_row(geocoder, row_data):
        """Geocode a single address."""
        row_index, address = row_data
        
        # Add delay to respect API rate limits
        time.sleep(DELAY_BETWEEN_REQUESTS)
        
        try:
            if not address or pd.isna(address) or address.strip() == '':
                return row_index, None, None, "Empty address"
            
            lat, lon = geocoder.geocode(address)
            
            with lock:
                stats['successful_geocodes'] += 1
            
            logger.info(f"Successfully geocoded row {row_index}: {address[:50]}... -> ({lat}, {lon})")
            return row_index, lat, lon, "Success"
            
        except GoogleUtilityError as e:
            with lock:
                stats['failed_geocodes'] += 1
            
            logger.warning(f"Failed to geocode row {row_index}: {address[:50]}... - {str(e)}")
            return row_index, None, None, f"Geocoding error: {str(e)}"
            
        except Exception as e:
            with lock:
                stats['failed_geocodes'] += 1
            
            logger.error(f"Unexpected error geocoding row {row_index}: {address[:50]}... - {str(e)}")
            return row_index, None, None, f"Unexpected error: {str(e)}"
    
    def log_final_stats():
        """Log final processing statistics."""
        duration = stats['end_time'] - stats['start_time']
        
        logger.info("=" * 50)
        logger.info("GEOCODING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total rows processed: {stats['total_rows']}")
        logger.info(f"Rows with existing coordinates: {stats['rows_with_coords']}")
        logger.info(f"Rows that needed geocoding: {stats['rows_to_geocode']}")
        logger.info(f"Successful geocodes: {stats['successful_geocodes']}")
        logger.info(f"Failed geocodes: {stats['failed_geocodes']}")
        logger.info(f"Total processing time: {duration:.2f} seconds")
        
        if stats['rows_to_geocode'] > 0:
            success_rate = (stats['successful_geocodes'] / stats['rows_to_geocode']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            if duration > 0:
                rate = stats['rows_to_geocode'] / duration
                logger.info(f"Processing rate: {rate:.2f} addresses/second")
        
        logger.info("=" * 50)
    
    try:
        stats['start_time'] = time.time()
        
        # Validate input file exists
        if not Path(INPUT_FILE).exists():
            raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found")
        
        logger.info(f"Starting to process CSV file: {INPUT_FILE}")
        logger.info(f"Using {MAX_WORKERS} parallel workers")
        
        # Initialize the utility
        geocoder = GoogleUtility(api_key=API_KEY)
        
        # Read CSV file
        try:
            df = pd.read_csv(INPUT_FILE)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
        
        # Validate required columns
        required_columns = ['combined', 'latitude_precisely', 'longitude_precisely']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        stats['total_rows'] = len(df)
        logger.info(f"Loaded CSV with {stats['total_rows']} rows")
        
        # Identify rows that need geocoding
        needs_geocoding_mask = df.apply(needs_geocoding, axis=1)
        rows_to_geocode = df[needs_geocoding_mask]
        
        stats['rows_with_coords'] = len(df) - len(rows_to_geocode)
        stats['rows_to_geocode'] = len(rows_to_geocode)
        
        logger.info(f"Rows with existing coordinates: {stats['rows_with_coords']}")
        logger.info(f"Rows needing geocoding: {stats['rows_to_geocode']}")
        
        if stats['rows_to_geocode'] == 0:
            logger.info("No rows need geocoding. Saving original data to output file.")
            df.to_csv(OUTPUT_FILE, index=False)
        else:
            # Create a copy of the dataframe to modify
            result_df = df.copy()
            
            # Prepare data for parallel processing
            geocoding_tasks = [
                (idx, row['combined']) 
                for idx, row in rows_to_geocode.iterrows()
            ]
            
            # Process in parallel
            logger.info("Starting parallel geocoding...")
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(geocode_single_row, geocoder, task): task 
                    for task in geocoding_tasks
                }
                
                # Process completed tasks
                completed_count = 0
                for future in as_completed(future_to_task):
                    row_index, lat, lon, status = future.result()
                    
                    # Update the result dataframe
                    if lat is not None and lon is not None:
                        result_df.at[row_index, 'latitude_precisely'] = lat
                        result_df.at[row_index, 'longitude_precisely'] = lon
                    
                    completed_count += 1
                    
                    # Log progress every 100 completed tasks
                    if completed_count % 100 == 0 or completed_count == len(geocoding_tasks):
                        progress = (completed_count / len(geocoding_tasks)) * 100
                        logger.info(f"Progress: {completed_count}/{len(geocoding_tasks)} ({progress:.1f}%)")
            
            # Save the result
            logger.info(f"Saving results to: {OUTPUT_FILE}")
            result_df.to_csv(OUTPUT_FILE, index=False)
        
        # Calculate final statistics
        stats['end_time'] = time.time()
        log_final_stats()
        
        logger.info("Geocoding process completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
    except ValueError as e:
        logger.error(f"Data error: {e}")
    except GoogleUtilityError as e:
        logger.error(f"GoogleUtility error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
