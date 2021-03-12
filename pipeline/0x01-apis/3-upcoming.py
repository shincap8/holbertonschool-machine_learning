#!/usr/bin/env python3
"""script that displays the upcoming launch with these information"""
import sys
import requests as rq
import time


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = rq.get(url)
    d_unix = int(time.time())
    for i in range(len(r.json())):
        if d_unix > r.json()[i]['date_unix']:
            d_unix = r.json()[i]['date_unix']
            ind = i
    launch_name = r.json()[ind]['name']
    date = r.json()[ind]['date_local']
    rocket_id = r.json()[ind]['rocket']
    launchpad_id = r.json()[ind]['launchpad']
    url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    r = rq.get(url)
    rocket_name = r.json()['name']
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)
    r = rq.get(url)
    launchpad_name = r.json()['name']
    launchpad_locality = r.json()['locality']
    print("{} ({}) {} - {} ({})".format(launch_name, date, rocket_name,
                                        launch_name, launchpad_locality))
