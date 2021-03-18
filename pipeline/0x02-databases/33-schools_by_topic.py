#!/usr/bin/env python3
"""Method schools by topics"""


def schools_by_topic(mongo_collection, topic):
    """function that returns the list of school having a specific topic
    Parameters:
        mongo_collection - will be the pymongo collection object
        topic - (string) will be topic searched
    """
    specific = []
    results = mongo_collection.find({"topics": {"$all": [topic]}})
    for result in results:
        specific.append(result)
    return specific
