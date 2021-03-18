#!/usr/bin/env python3
"""Method update topics"""


def update_topics(mongo_collection, name, topics):
    """function that changes all topics of a school document based
        on the name
    Parameters:
        mongo_collection - will be the pymongo collection object
        name - (string) will be the school name to update
        topics - (list of strings) will be the list of topics approached
            in the school
    """
    nvalues = {"$set": {"topics": topics}}
    mongo_collection.update_many({"name": name}, nvalues)
