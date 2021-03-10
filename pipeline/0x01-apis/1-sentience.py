#!/usr/bin/env python3
"""sentientPlanets method"""
import requests as rq


def sentientPlanets():
    """Function returns the list of names of
    the home planets of all sentient species"""
    planets = []
    page = 1
    state = True
    while state:
        url = "https://swapi-api.hbtn.io/api/species/?page=" + str(page)
        r = rq.get(url)
        data = r.json()
        results = data['results']
        for specie in results:
            if specie['classification'] == 'sentient' or \
               specie['designation'] == 'sentient':
                homeworld = specie['homeworld']
                if homeworld is not None:
                    req = rq.get(specie['homeworld'])
                    Data = req.json()
                    planets.append(Data['name'])
        if data['next'] is None:
            state = False
        page += 1
    return planets
