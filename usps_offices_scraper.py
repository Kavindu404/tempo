#!/usr/bin/env python3
"""
USPS Post Office Locations Scraper

This script provides multiple approaches to get USPS office addresses:
1. Google Places API (recommended - requires API key)
2. Web scraping USPS store locator
3. OpenStreetMap Overpass API (free alternative)

Output: CSV file with ZIP_CODE and ADDRESS columns
"""

import requests
import csv
import json
import time
import re
from urllib.parse import urlencode, quote
from typing import List, Dict, Optional
import zipfile
import io

class USPSLocationScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.results = []
    
    def method1_google_places_api(self, api_key: str, states: Optional[List[str]] = None) -> List[Dict]:
        """
        Method 1: Use Google Places API to find USPS offices
        Requires Google Places API key (paid service after free tier)
        """
        print("Using Google Places API...")
        
        if not states:
            # US state abbreviations
            states = [
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
            ]
        
        offices = []
        base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        
        for state in states:
            print(f"Searching {state}...")
            query = f"USPS Post Office {state}"
            
            params = {
                'query': query,
                'key': api_key,
                'type': 'post_office'
            }
            
            try:
                response = self.session.get(base_url, params=params)
                data = response.json()
                
                if data.get('status') == 'OK':
                    for place in data.get('results', []):
                        # Get detailed info for each place
                        place_id = place.get('place_id')
                        if place_id:
                            details = self._get_place_details(place_id, api_key)
                            if details:
                                office = self._extract_office_data(details)
                                if office:
                                    offices.append(office)
                
                # Handle pagination
                while 'next_page_token' in data:
                    time.sleep(2)  # Required delay for next_page_token
                    params['pagetoken'] = data['next_page_token']
                    response = self.session.get(base_url, params=params)
                    data = response.json()
                    
                    if data.get('status') == 'OK':
                        for place in data.get('results', []):
                            place_id = place.get('place_id')
                            if place_id:
                                details = self._get_place_details(place_id, api_key)
                                if details:
                                    office = self._extract_office_data(details)
                                    if office:
                                        offices.append(office)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error searching {state}: {e}")
                continue
        
        return offices
    
    def _get_place_details(self, place_id: str, api_key: str) -> Optional[Dict]:
        """Get detailed place information from Google Places API"""
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'place_id': place_id,
            'key': api_key,
            'fields': 'name,formatted_address,address_components,types'
        }
        
        try:
            response = self.session.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'OK':
                return data.get('result')
        except Exception as e:
            print(f"Error getting place details: {e}")
        
        return None
    
    def _extract_office_data(self, place_details: Dict) -> Optional[Dict]:
        """Extract office data from Google Places API response"""
        if not place_details:
            return None
        
        # Check if it's actually a USPS office
        name = place_details.get('name', '').lower()
        types = place_details.get('types', [])
        
        if not ('usps' in name or 'post office' in name or 'postal service' in name):
            return None
        
        if 'post_office' not in types and 'establishment' not in types:
            return None
        
        # Extract ZIP code from address components
        zip_code = None
        for component in place_details.get('address_components', []):
            if 'postal_code' in component.get('types', []):
                zip_code = component.get('long_name')
                break
        
        address = place_details.get('formatted_address', '')
        
        if zip_code and address:
            return {
                'zip_code': zip_code,
                'address': address,
                'name': place_details.get('name', '')
            }
        
        return None
    
    def method2_overpass_api(self) -> List[Dict]:
        """
        Method 2: Use OpenStreetMap Overpass API (free)
        Less comprehensive but doesn't require API key
        """
        print("Using OpenStreetMap Overpass API...")
        
        # Overpass query to find post offices in the US
        overpass_query = """
        [out:json][timeout:300];
        (
          node["amenity"="post_office"]["addr:country"="US"];
          way["amenity"="post_office"]["addr:country"="US"];
          relation["amenity"="post_office"]["addr:country"="US"];
        );
        out center meta;
        """
        
        url = "https://overpass-api.de/api/interpreter"
        
        try:
            response = self.session.post(url, data=overpass_query)
            data = response.json()
            
            offices = []
            for element in data.get('elements', []):
                office = self._extract_osm_office_data(element)
                if office:
                    offices.append(office)
            
            return offices
        
        except Exception as e:
            print(f"Error with Overpass API: {e}")
            return []
    
    def _extract_osm_office_data(self, element: Dict) -> Optional[Dict]:
        """Extract office data from OpenStreetMap element"""
        tags = element.get('tags', {})
        
        # Check if it's a USPS office
        name = tags.get('name', '').lower()
        operator = tags.get('operator', '').lower()
        
        if not ('usps' in name or 'usps' in operator or 'united states postal service' in name):
            return None
        
        # Extract address components
        house_number = tags.get('addr:housenumber', '')
        street = tags.get('addr:street', '')
        city = tags.get('addr:city', '')
        state = tags.get('addr:state', '')
        zip_code = tags.get('addr:postcode', '')
        
        # Build full address
        address_parts = []
        if house_number and street:
            address_parts.append(f"{house_number} {street}")
        elif street:
            address_parts.append(street)
        
        if city:
            address_parts.append(city)
        
        if state:
            address_parts.append(state)
        
        if zip_code:
            address_parts.append(zip_code)
        
        if zip_code and len(address_parts) >= 3:  # At least street, city, state, zip
            return {
                'zip_code': zip_code,
                'address': ', '.join(address_parts),
                'name': tags.get('name', 'USPS Post Office')
            }
        
        return None
    
    def method3_scrape_usps_locator(self, sample_zip_codes: List[str] = None) -> List[Dict]:
        """
        Method 3: Scrape USPS store locator
        Note: This is for educational purposes and should respect rate limits
        """
        print("Scraping USPS store locator...")
        
        if not sample_zip_codes:
            # Sample ZIP codes from major cities across all states
            sample_zip_codes = [
                '10001', '90210', '60601', '77001', '85001', '80201', '06101',
                '19701', '32801', '30301', '96801', '83701', '62701', '46201',
                '50301', '66101', '40201', '70112', '04101', '21201', '02101',
                '48201', '55401', '39201', '63101', '59601', '68501', '89101',
                '03301', '07101', '87101', '10001', '27601', '58501', '43201',
                '73101', '97201', '15201', '02901', '29201', '57101', '37201',
                '73301', '84101', '05601', '23219', '98101', '25301', '53201',
                '82001'
            ]
        
        offices = []
        
        for zip_code in sample_zip_codes:
            try:
                print(f"Searching near ZIP {zip_code}...")
                
                # USPS store locator API endpoint
                url = "https://tools.usps.com/UspsToolsRestUser/rest/POLocator/findLocations"
                
                params = {
                    'address': zip_code,
                    'productType': 'All',
                    'radius': '50'  # 50 mile radius
                }
                
                response = self.session.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for location in data.get('locations', []):
                        office = self._extract_usps_locator_data(location)
                        if office:
                            offices.append(office)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error searching ZIP {zip_code}: {e}")
                continue
        
        return offices
    
    def _extract_usps_locator_data(self, location: Dict) -> Optional[Dict]:
        """Extract office data from USPS locator response"""
        try:
            zip_code = location.get('zip5', '')
            
            # Build address
            address_parts = []
            
            if location.get('address'):
                address_parts.append(location['address'])
            
            if location.get('city'):
                address_parts.append(location['city'])
            
            if location.get('state'):
                address_parts.append(location['state'])
            
            if zip_code:
                address_parts.append(zip_code)
            
            if zip_code and len(address_parts) >= 3:
                return {
                    'zip_code': zip_code,
                    'address': ', '.join(address_parts),
                    'name': location.get('name', 'USPS Post Office')
                }
        
        except Exception as e:
            print(f"Error extracting location data: {e}")
        
        return None
    
    def save_to_csv(self, offices: List[Dict], filename: str = 'usps_offices.csv'):
        """Save office data to CSV file"""
        print(f"Saving {len(offices)} offices to {filename}...")
        
        # Remove duplicates based on ZIP code and address
        unique_offices = {}
        for office in offices:
            key = f"{office['zip_code']}_{office['address']}"
            if key not in unique_offices:
                unique_offices[key] = office
        
        offices = list(unique_offices.values())
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['ZIP_CODE', 'ADDRESS']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for office in offices:
                writer.writerow({
                    'ZIP_CODE': office['zip_code'],
                    'ADDRESS': office['address']
                })
        
        print(f"Saved {len(offices)} unique offices to {filename}")

def main():
    scraper = USPSLocationScraper()
    all_offices = []
    
    print("USPS Post Office Locations Scraper")
    print("=" * 40)
    
    # Method 1: Google Places API (commented out - requires API key)
    # google_api_key = "YOUR_GOOGLE_PLACES_API_KEY"
    # if google_api_key != "YOUR_GOOGLE_PLACES_API_KEY":
    #     print("Method 1: Google Places API")
    #     offices = scraper.method1_google_places_api(google_api_key)
    #     all_offices.extend(offices)
    #     print(f"Found {len(offices)} offices via Google Places API\n")
    
    # Method 2: OpenStreetMap Overpass API (free)
    print("Method 2: OpenStreetMap Overpass API")
    offices = scraper.method2_overpass_api()
    all_offices.extend(offices)
    print(f"Found {len(offices)} offices via OpenStreetMap\n")
    
    # Method 3: USPS Locator (sample approach)
    print("Method 3: USPS Store Locator (sample)")
    offices = scraper.method3_scrape_usps_locator()
    all_offices.extend(offices)
    print(f"Found {len(offices)} offices via USPS locator\n")
    
    if all_offices:
        scraper.save_to_csv(all_offices)
        print(f"Total offices collected: {len(all_offices)}")
    else:
        print("No offices found. Try enabling Google Places API method for better results.")

if __name__ == "__main__":
    main()
