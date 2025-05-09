def doc_to_text(doc) -> str:
    """
    Question: <question>
    Choices:
    A. <option1>
    B. <option2>
    C. <option3>
    D. <option4>
    Answer:
    """
    choices = [doc["option1"], doc["option2"], doc["option3"], doc["option4"]]
    option_choices = {
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
    }

    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"

    return prompt


def doc_to_target(doc) -> int:
    """
    Returns the index of the correct answer in the list of choices
    """
    target = doc["target"]

    if target == "option1":
        return "A"
    elif target == "option2":
        return "B"
    elif target == "option3":
        return "C"
    elif target == "option4":
        return "D"
    else:
        raise ValueError(f"Invalid option {target}")


def process_results_gen(doc, results):
    candidate = results[0]

    target = doc["target"]

    if target == "option1":
        gold = "A"
    elif target == "option2":
        gold = "B"
    elif target == "option3":
        gold = "C"
    elif target == "option4":
        gold = "D"
    else:
        raise ValueError(f"Invalid option {target}")

    candidate = candidate.strip().lower().split("\n")[0].split(" ")[0].strip()

    if "." in candidate:
        candidate = candidate.split(".")[0]

    retval = 0
    if candidate == gold:
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results
