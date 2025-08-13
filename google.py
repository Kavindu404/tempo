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
