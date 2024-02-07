from classes import Dataset

basketball = Dataset(
    name="Basketball",
    keywords=["Basketball"],
    allow_list=[
        "1991107619818",
        "19753803125",
        "19753803009",
        "199110766513",
        "1991107619129",
        "1991107616835",
        "19753803184",
        "1991107619479",
        "19753803183",
        "1991107619410",
        "199110767086",
        "1991107612031",
        "1991107619686",
        "1991107619798",
        "1991107619799",
        "19753803046",
        "19753803057",
        "19753803081",
        "19753803041",
        "1991107610782",
        "199110767085",
        "19753803070",
    ],
    block_list=[],
)
baseball = Dataset(
    name="Baseball",
    keywords=[
        "baseball",
        "ball team",
        "ball players",
        "cardinals",
        "ball tournament",
        "white sox",
        "lethbridge north stars",
        "ball park",
        "ball field",
        "fastball team" "hardball team",
        "Little League",
    ],
    allow_list=[
        "199110769122",
        "199110768824",
        "199110767797",
        "1991107617706",
        "19753802006",
        "199110769125",
        "1991107618048",
        "19753802084",
        "199110767502",
        "19753802135",
        "1991107617705",
        "19753802081",
        "1991107618205",
        "19753802090",
        "19753802082",
        "19753802097",
        "19753802101",
        "19752311054",
    ],
    block_list=[],
)
football = Dataset(
    name="Football",
    keywords=[
        "quarterback",
        "football",
        "touchdown",
        "interception",
        "sack",
        "fumble",
        "punt",
        "hail mary",
        "snap",
        "end zone",
        "field goal",
    ],
    allow_list=[],
    block_list=[],
)
soccer = Dataset(
    name="Soccer",
    keywords=["soccer", "European football"],
    allow_list=[],
    block_list=[],
)
women = Dataset(
    name="Women",
    keywords=["woman", "women", "female", "wife", "queen", "she"],
    allow_list=[],
    block_list=[],
)
men = Dataset(
    name="Men",
    keywords=["man", "men", "male", "husband", "king", "he"],
    allow_list=[],
    block_list=[],
)
cars = Dataset(
    name="Auto",
    keywords=[
        "auto",
        "car",
        "cars",
        "vehicle",
        "vehicles",
        "automobile",
        "automobiles",
        "parking lot",
        "tire",
        "truck",
        "train-truck",
        "tractor",


        # Car brands
        "Ford",
        "Chrysler",
        "Duesenberg",
        "Chevrolet",
        "Chevy",
        "Willys-Overland",
        "Tucker",
        "Oldsmobile",
        "Cadillac",
        "Mustang",
        "Camaro",
        "Pontiac",
        "AMC",
        "Dodge",
        "Buick",
        "Alfa Romeo",
        "Jaguar",
        "British Leyland",
        "Ferrari",
        "Porsche",
        "BMW",
        "Plymouth",
        "Mercedes-Benz",
        "Lotus",
        "Datsun",
        "Lamborghini",
        "Volvo",
        "Studebaker",
        "Jeep",
        "Road Patrol",


    ],
    allow_list=[
        "1991107618405",
        "19752608087b",
        "199110767246",
        "199110768726",
        "199110769073",
        "1991107617414",
        "199110761890",
        "1991107619903",
        "199110768722",
        "19752303298",
        "19752303223",
        "199110765469",
        "199110769074",
        "19752606016",
        "199110767324",
        "199110769072",
        "199110766700",
        "199110769713",
        "19753500006",
        "19752390781"
    ],
    block_list=[],
)
indian = Dataset(
    name="Indian",
    keywords=[
        "indian",
        "native",
        "first nations",
        "chief",
        # "ceremony",
        "Indigenous",
        "tribe",
        "Kainai",
        "blood tribe",
        "traditional dress",
        "Blood Reserve",
        "headdress",
        "Red Bull Society",
    ],
    allow_list=[
        "1991107618702",
        "1991107618703",
        "19754011095",
    ],
    block_list=[
        "199110765217"
    ],
)

datasets: list[Dataset] = [
    basketball,
    baseball,
    football,
    soccer,
    indian,
    women,
    men,
    cars,
]
