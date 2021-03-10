#!/usr/bin/env python3
"""availableShips method"""
import requests as rq


def availableShips(passengerCount):
    """method that returns the list of ships
    that can hold a given number of passengers"""
    vehicles = []
    page = 1
    state = True
    while state:
        url = "https://swapi-api.hbtn.io/api/starships/?page=" + str(page)
        r = rq.get(url)
        data = r.json()
        results = data['results']
        for vehicle in results:
            passenger = vehicle['passengers']
            passenger = passenger.replace(',', "")
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                vehicles.append(vehicle['name'])
        if data['next'] is None:
            state = False
        page += 1
    return vehicles
