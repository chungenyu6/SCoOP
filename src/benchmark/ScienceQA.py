from datasets import load_dataset


class ScienceQA:
    def __init__(self):
        self.ds = load_dataset("derek-thomas/ScienceQA")

    def obtain_size(self):
        return len(self.ds["test"])

    def retrieve(self, idx):
        row = self.ds["test"][idx]
        question = row["question"]
        lecture = row["lecture"]
        hint = row["hint"]

        hint_str = f"HINT: {hint}\\n" if hint else ""
        lecture_str = f"LECTURE: {lecture}\\n" if lecture else ""

        question += "\\n"
        question += hint_str
        question += lecture_str

        choices = ""
        choice_numbers = ""
        for i, choice in enumerate(row["choices"]):
            choices += f"({i}): {choice}\\n"
            choice_numbers += f"{i}, "
        choice_numbers = choice_numbers[:-2]

        question += choices
        question += "\\n"
        question += (
            f"This is a single choice question, answer only with choice number in {choice_numbers}."
        )

        result = {
            "idx": idx,
            "img": row["image"],
            "question": question,
            "label": row["answer"],
            "num_c": len(row["choices"]),
        }
        return result


if __name__ == "__main__":
    benchmark = ScienceQA()
    print(benchmark.retrieve(0))
