import json
from pathlib import Path

DOCS = {
    "python_snake.txt": "The python is a large snake that kills its prey by squeezing. Pythons live in warm forests and swamps and eat birds and small mammals.",
    "python_code.txt": "Python is a popular programming language for data analysis and web development. Its simple syntax makes Python easy for beginners to read and write.",
    "apple_fruit.txt": "An apple is a sweet fruit that grows on trees in orchards. Apples are rich in fiber and are often baked into pies or pressed into juice.",
    "apple_company.txt": "Apple is a technology company that designs phones, laptops and watches. The company sells its devices through stores and an online shop.",
    "jaguar_animal.txt": "The jaguar is a big cat that lives in the rainforests of South America. Jaguars are strong swimmers and hunt turtles, deer and fish.",
    "jaguar_car.txt": "Jaguar is a British car maker known for luxury sedans and sports cars. The company builds its cars with powerful engines and leather interiors.",
    "bank_money.txt": "A bank is a business that keeps your money safe and gives loans. Banks charge interest on loans and pay interest on savings accounts.",
    "bank_river.txt": "The bank of a river is the sloping land along its edge. River banks can erode during floods and are often home to reeds and frogs.",
    "mercury_planet.txt": "Mercury is the smallest planet and the closest to the sun. A year on Mercury lasts only eighty-eight days on Earth.",
    "mercury_metal.txt": "Mercury is a silvery metal that is liquid at room temperature. It was used in old thermometers but is poisonous, so it is now avoided.",
}

QUESTIONS = [
    ("how does a python kill its prey", "python_snake.txt"),
    ("is python easy for beginners", "python_code.txt"),
    ("are apples good for baking pies", "apple_fruit.txt"),
    ("where does the company sell its phones and laptops", "apple_company.txt"),
    ("which big cat swims and hunts turtles", "jaguar_animal.txt"),
    ("which British company makes luxury sports cars", "jaguar_car.txt"),
    ("how does a bank earn money from loans", "bank_money.txt"),
    ("what happens to a river bank during a flood", "bank_river.txt"),
    ("how long is a year on the closest planet to the sun", "mercury_planet.txt"),
    ("why is liquid silver metal poisonous", "mercury_metal.txt"),
]

folder = Path("data/eval_corpus")
folder.mkdir(parents=True, exist_ok=True)
for name, text in DOCS.items():
    (folder / name).write_text(text, encoding="utf-8")

dataset = [{"question": q, "expected_source": s} for q, s in QUESTIONS]
Path("evaluation/questions_big.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
print(f"Wrote {len(DOCS)} documents and {len(dataset)} questions.")