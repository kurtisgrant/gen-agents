import json

class EnvNode:
    def __init__(self, name, children=None, state=None):
        self.name = name
        self.children = children if children is not None else []
        self.state = state

    def to_dict(self):
        return {
            "name": self.name,
            "children": [child.to_dict() for child in self.children],
            "state": self.state
        }
    
    def print(self):
        print(json.dumps(self.to_dict(), indent=1))

    def get_json(self):
        return json.dumps(self.to_dict())
    
    # Method for getting a node by name
    def get_node(self, name):
        if self.name == name:
            return self
        else:
            for child in self.children:
                node = child.get_node(name)
                if node:
                    return node
            return None
        
    # Method for getting a node's parent by name
    def get_parent(self, name):
        for child in self.children:
            if child.name == name:
                return self
            else:
                node = child.get_parent(name)
                if node:
                    return node
        return None
    
    def validate_travel(self, current_location, destination):
        if current_location == destination:
            return False
        if self.get_node(current_location) is None:
            return False
        if self.get_node(destination) is None:
            return False
        # If current location is not a parent or child of destination, return False
        if self.get_parent(current_location) is None and self.get_node(current_location) not in self.get_node(destination).children:
            return False
        return True

    @staticmethod
    def from_dict(dict_obj):
        return EnvNode(
            name=dict_obj["name"],
            children=[EnvNode.from_dict(child) for child in dict_obj.get("children", [])],
            state=dict_obj.get("state")
        )
    
    @staticmethod
    def init():
        return EnvNode("Village", [
            EnvNode("Burt's House", [
                EnvNode("Burt's Bedroom", [
                    EnvNode("Bed", state="Made"),
                    EnvNode("Closet", state="Open"),
                    EnvNode("Desk", state="Tidy")
                ]),
                EnvNode("Living Room", [
                    EnvNode("Sofa", state="Occupied"),
                    EnvNode("TV", state="On")
                ]),
                EnvNode("Kitchen", [
                    EnvNode("Refrigerator", state="Cold"),
                    EnvNode("Stove", state="Off"),
                    EnvNode("Coffee Pot", state="Brewing...")
                ]),
                EnvNode("Bathroom", [
                    EnvNode("Sink", state="Clean"),
                    EnvNode("Toilet", state="Flushed")
                ])
            ]),
            EnvNode("Town Square", [
                EnvNode("Store", [
                    EnvNode("Aisles", [
                        EnvNode("Food Products", state="Stocked"),
                        EnvNode("Hardware Products", state="Stocked")
                    ]),
                    EnvNode("Checkout", [
                        EnvNode("Payment Terminal", state="Idle"),
                        EnvNode("Conveyor Belt", state="Empty")
                    ])
                ]),
                EnvNode("Pub", [
                    EnvNode("Seating Area", [
                        EnvNode("Tables", [
                            EnvNode("Table 1", state="Occupied"),
                            EnvNode("Table 2", state="Empty")
                        ])
                    ]),
                    EnvNode("Games Room", [
                        EnvNode("Pool Table", state="In use"),
                        EnvNode("Dartboard", state="Available")
                    ]),
                    EnvNode("Bar Counter", [
                        EnvNode("Cash Register", state="Idle"),
                        EnvNode("Drinks", state="Available")
                    ])
                ]),
                EnvNode("Seating Area", [
                    EnvNode("Benches", [
                        EnvNode("Bench 1", state="Occupied"),
                        EnvNode("Bench 2", state="Empty")
                    ])
                ])
            ]),
            EnvNode("Park", [
                EnvNode("Nature Trail", [
                    EnvNode("Lookout Point", [
                        EnvNode("Binoculars", state="Available"),
                        EnvNode("Information Board", state="Updated")
                    ])
                ]),
                EnvNode("Flower Garden", [
                    EnvNode("Roses", state="Blooming"),
                    EnvNode("Tulips", state="Wilting")
                ])
            ])
        ])