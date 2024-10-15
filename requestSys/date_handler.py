import sys
import requests
import json
import datetime
import os 
import time

def distribute_data():

    def get_weekend_dates(year, month):
        """
        Get a list of weekend dates (Saturdays and Sundays) for a given month and year.

        Parameters:
        year (int): The year for which to find weekend dates.
        month (int): The month for which to find weekend dates.

        Returns:
        list: A list of datetime objects representing the weekend dates in the specified month.
        """
        weekend_dates = []
        first_day = datetime.datetime(year, month, 1)  # Get the first day of the month
        if month == 12:
            next_month = datetime.datetime(year + 1, 1, 1)  # Handle December to January transition
        else:
            next_month = datetime.datetime(year, month + 1, 1)  # Get the first day of the next month
        num_days = (next_month - first_day).days  # Calculate the number of days in the month
        for day in range(1, num_days + 1):
            current_date = datetime.datetime(year, month, day)  # Create a date object for each day
            if current_date.weekday() >= 5 and current_date >= datetime.datetime.now():  # Check if the day is Saturday (5) or Sunday (6) and not in the past
                weekend_dates.append(current_date)  # Add weekend dates to the list

        # Write out the weekends to a .json file    
        with open('weekend_dates.json', 'w') as file:
            json.dump(weekend_dates, file, default=str, indent=4)
        print(f'Weekend dates saved to weekend_dates.json')
        

        return weekend_dates  # Return the list of weekend dates

    def distribute_requests(weekend_dates, total_requests=500):
        """
        Distribute a specified number of requests across weekend dates.

        Parameters:
        weekend_dates (list): A list of weekend dates (datetime objects) to distribute requests over.
        total_requests (int): The total number of requests to distribute (default is 500).

        Returns:
        dict: A dictionary where the keys are weekend dates and the values are the number of requests scheduled for each date.
        """
        num_weekend_days = len(weekend_dates)  # Count the number of weekend days
        base_requests = total_requests // num_weekend_days  # Calculate the base number of requests per weekend day
        extra_requests = total_requests % num_weekend_days  # Calculate any extra requests to distribute

        requests_schedule = {}  # Initialize a dictionary to hold the requests schedule
        for i, date in enumerate(sorted(weekend_dates)):  # Iterate over the sorted weekend dates
            # Calculate the number of requests for today, adding one if there are extra requests to distribute
            requests_today = base_requests + (1 if i < extra_requests else 0)
            # Convert date to string for JSON serialization
            requests_schedule[date.date().isoformat()] = {
                'requests': requests_today,
                'interval': 24 * 60 * 60 // requests_today if requests_today > 0 else 0  # seconds in a day divided by requests
            }
        
        # Write the weekend dates to weekend_dates.json, including the extra requests information
        with open('weekend_dates.json', 'w') as file:
            json.dump({
                'weekend_dates': [date.isoformat() for date in weekend_dates],
                'requests_schedule': requests_schedule
            }, file, indent=4)  # Add 'file' as the second argument
        print(f'Weekend dates and requests schedule saved to weekend_dates.json')

        return requests_schedule  # Return the completed requests schedule
    
    today = datetime.datetime.now().date()
    year = today.year
    month = today.month

    weekend_dates = get_weekend_dates(year, month)
    request_schedule = distribute_requests(weekend_dates)

    print("[+] written weeken dates and the written request schedule")
    return request_schedule

distribute_data()
