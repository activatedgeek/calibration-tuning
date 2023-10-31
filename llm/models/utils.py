def getchainattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes:
        obj = getattr(obj, a)
    return obj


def delchainattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    try:
        delattr(obj, attributes[-1])
    except AttributeError:
        print(obj)
        raise


def setchainattr(obj, attr, value):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attributes[-1], value)
