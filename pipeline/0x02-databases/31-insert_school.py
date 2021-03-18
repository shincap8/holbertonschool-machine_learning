#!/usr/bin/env python3
"""Method insert school"""


def insert_school(mongo_collection, **kwargs):
    """Method that inserts a new document in a collection based on kwargs:
    Parameters:
        mongo_collection will be the pymongo collection object
    Returns:
        the new _id
    """
    _id = mongo_collection.insert_one(kwargs).inserted_id
    return _id
