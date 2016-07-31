import os
import json
from pymongo import MongoClient, errors


class MongoConnection():
    def __init__(self):
        pass


    @staticmethod
    def default_connection():
        """
        Reads the connection information from a local configuration/override.json if exists, otherwise will read the
        configuration/default.json

        Sets just the values
        :return: the MongoDB connection already initialized
        """
        default = "configuration/default.json"
        override = "configuration/override.json"
        with open(default, 'rb') as default_file:
            values = json.load(default_file)
            if os.path.exists(override):
                with open(override, 'rb') as override_file:
                    values.update(json.load(override_file))

        # Now values contain the override values
        return MongoClient(**values)

