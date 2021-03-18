#!/usr/bin/env python3
"""Method List all"""


def list_all(mongo_collection):
    """Method that lists all documents in a collection:
    Parameters:
        mongo_collection - the pymongo collection object
    Return:
        empty list if no document in the collection
    """
    documents = []
    col = mongo_collection.find()
    for doc in col:
        documents.append(doc)
    return documents
