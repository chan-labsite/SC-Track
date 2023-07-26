import logging


class InsertError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Insert Error: {self.value}!"

    def __repr__(self):
        return self.__str__()


class MitosisError(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"cannot match sister cells! {self.message if self.message else ''}!"

    def __repr__(self):
        return self.__str__()


class ErrorMatchMitosis(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"Unreasonable cell size, should not happen cell division! {self.message if self.message else ''}!"

    def __repr__(self):
        return self.__str__()


class NodeExistError(Exception):
    def __init__(self, node, message=None):
        self.message = message
        self.node = node

    def __str__(self):
        return f"cannot add node: {self.node}, already exists! {self.message if self.message else ''}!"

    def __repr__(self):
        return self.__str__()


class MatchFailed(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"match failed! {self.message if self.message else ''}!"

    def __repr__(self):
        return self.__str__()


class StatusError(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        if self.message:
            return self.message
        else:
            return "when set the status, some error occurred..."

    def __repr__(self):
        return self.__str__()
