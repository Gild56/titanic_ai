import pandas as pd

import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from colorama import Fore, Style, init


init(autoreset=True)

red = Fore.RED
blue = Fore.BLUE
green = Fore.GREEN
cyan = Fore.CYAN
yellow = Fore.YELLOW
reset = Style.RESET_ALL


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


df = pd.read_csv(resource_path("titanic.csv"))

df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

df["Embarked"].fillna("S", inplace=True)

ages = df.groupby(by="Pclass")["Age"].median()
age_1 = ages[1]
age_2 = ages[2]
age_3 = ages[3]


def fill_age(row):
    if pd.isnull(row["Age"]):
        if row["Pclass"] == 1:
            return age_1
        elif row["Pclass"] == 2:
            return age_2
        else:
            return age_3
    else:
        return row["Age"]


df["Age"] = df.apply(fill_age, axis=1)


def fill_sex(data):
    return 1 if data == "male" else 0


df["Sex"] = df["Sex"].apply(fill_sex)

df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

x = df.drop("Survived", axis=1)
y = df["Survived"]

sc = StandardScaler()
x = sc.fit_transform(x)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(x, y)

features = list(df.drop("Survived", axis=1).columns)


def get_embarked_vars(city_num):
    if city_num == 1:
        return (1, 0, 0)  # Southampton
    elif city_num == 2:
        return (0, 1, 0)  # Cherbourg
    return (0, 0, 1)  # Queenstown


def predict_survival(
        pclass, sex, age, sibsp, parch, embarked_c, embarked_q, embarked_s):
    mean_prices = [0, 84, 20, 13]
    fare = mean_prices[pclass]
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked_C": [embarked_c],
        "Embarked_Q": [embarked_q],
        "Embarked_S": [embarked_s]
    })

    input_data = input_data[features]

    input_data = sc.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return True
    return False


def print_person_info(
        pclass, sex, age, sibsp, parch, embarked_c, embarked_q, result, lang):
    str(age)
    if sex == 1:
        if lang == 'ru':
            sex_description = 'мужчиной'
        elif lang == 'ua':
            sex_description = 'чоловіком'
        elif lang == 'fr':
            sex_description = 'un homme'
        else:
            sex_description = 'a male'
    else:
        if lang == 'ru':
            sex_description = 'женщиной'
        elif lang == 'ua':
            sex_description = 'жінкою'
        elif lang == 'fr':
            sex_description = 'une femme'
        else:
            sex_description = 'a female'

    if embarked_c:
        city = "Шербура" if lang == 'ru' or lang == 'ua' else "Cherbourg"
    elif embarked_q:
        city = "Квинстауна" if lang == 'ru' or lang == 'ua' else "Queenstown"
    else:
        city = "Саутгемптона" if lang == 'ru' or lang == 'ua' else "Southampton"

    if result:
        if lang == 'ru':
            result_text = "выжил"
        elif lang == 'ua':
            result_text = "пережи"
        elif lang == 'fr':
            result_text = "aurait survécu"
        else:
            result_text = "would survive"
        result_text = f"{green}{result_text}"
    else:
        if lang == 'ru':
            result_text = "не выжил"
        elif lang == 'ua':
            result_text = "не пережи"
        elif lang == 'fr':
            result_text = "n'aurait pas survécu"
        else:
            result_text = "wouldn't survive"
        result_text = f"{red}{result_text}"

    description = (
        f"   Этот пассажир был {yellow}{sex_description}{reset} и "
        f"принадлежал к {yellow}{pclass}{reset} классу. "
        f"Ему было {yellow}{age}{reset} лет. У него было {yellow}{sibsp}{reset} "
        f"друзей и {yellow}{parch}{reset} родственников на борту. "
        f"Он отправился из {yellow}{city}{reset}. "
        f"По результатам модели, он бы {result_text}{reset}." if lang == 'ru' and sex == 1 else

        f"   Этот пассажир был {yellow}{sex_description}{reset} и "
        f"принадлежал к {yellow}{pclass}{reset} классу. "
        f"Емa было {yellow}{age}{reset} лет. У неё было {yellow}{sibsp}{reset} "
        f"друзей и {yellow}{parch}{reset} родственников на борту. "
        f"Она отправилась из {yellow}{city}{reset}. "
        f"По результатам модели, она бы {result_text}а{reset}." if lang == 'ru' and sex == 0 else


        f"   Ця особа була {yellow}{sex_description}{reset} та "
        f"належала до {yellow}{pclass}{reset} класу. "
        f"Його вік складав {yellow}{age}{reset} років. "
        f"У неї було {yellow}{sibsp}{reset} друзів "
        f"і {yellow}{parch}{reset} родичів на борту. "
        f"Він відправився з {yellow}{city}{reset}. "
        f"За результатами моделі, він би {result_text}в{reset}." if lang == 'ua' and sex == 1 else

        f"   Ця особа була {yellow}{sex_description}{reset} та "
        f"належала до {yellow}{pclass}{reset} класу. "
        f"Її вік складав {yellow}{age}{reset} років. "
        f"У неї було {yellow}{sibsp}{reset} друзів "
        f"і {yellow}{parch}{reset} родичів на борту. "
        f"Вона відправилась з {yellow}{city}{reset}. "
        f"За результатами моделі, вона би {result_text}ла{reset}." if lang == 'ua' and sex == 0 else


        f"   Ce passager était {yellow}{sex_description}{reset} "
        f"appartennant à la classe numéro {yellow}{pclass}{reset}. "
        f"Il avait {yellow}{age}{reset} ans. "
        f"Il avait {yellow}{sibsp}{reset} amis "
        f"et {yellow}{parch}{reset} membres de la famille à bord. "
        f"Il a embarqué à {yellow}{city}{reset}. "
        f"Selon le modèle, il {result_text}{reset}." if lang == 'fr' and sex == 1 else

        f"   Ce passager était {yellow}{sex_description}{reset} "
        f"appartennant à la classe numéro {yellow}{pclass}{reset}. "
        f"Elle avait {yellow}{age}{reset} ans. "
        f"Elle avait {yellow}{sibsp}{reset} amis "
        f"et {yellow}{parch}{reset} membres de la famille à bord. "
        f"Elle a embarqué à {yellow}{city}{reset}. "
        f"Selon le modèle, elle {result_text}{reset}." if lang == 'fr' and sex == 0 else


        f"   This passenger was a {yellow}{sex_description}{reset}, "
        f"belonging to class {yellow}{pclass}{reset}. "
        f"His age was {yellow}{age}{reset}. "
        f"He had {yellow}{sibsp}{reset} friends "
        f"and {yellow}{parch}{reset} family members aboard. "
        f"He embarked in {yellow}{city}{reset}. "
        f"According to the model, he {result_text}{reset}." if sex == 1 else

        f"   This passenger was a {yellow}{sex_description}{reset}, "
        f"belonging to class {yellow}{pclass}{reset}. "
        f"Her age was {yellow}{age}{reset}. "
        f"She had {yellow}{sibsp}{reset} friends "
        f"and {yellow}{parch}{reset} family members aboard. "
        f"She embarked in {yellow}{city}{reset}. "
        f"According to the model, she {result_text}{reset}."
    )

    print(f"\n{description}\n")


def get_user_input():
    global lang

    if lang == 'ru':
        messages = [
            "Введите класс каюты (1, 2 или 3): ",
            "Введите пол (1 для мужского, 0 для женского): ",
            "Введите возраст: ",
            "Введите количество друзей на борту: ",
            "Введите количество родственников на борту: ",
            "Выберите город отправления:\n1 - Саутгемптон\n2 - Шербур\n3 - Квинстаун",
            "Введите номер города (1, 2 или 3): ",
            "Хотите проверить ещё одного пассажира? (да/нет): "]

    elif lang == 'ua':
        messages = [
            "Введіть клас каюти (1, 2 або 3): ",
            "Введіть стать (1 для чоловіків, 0 для жінок): ",
            "Введіть вік: ",
            "Введіть кількість друзів на борту: ",
            "Введіть кількість родичів на борту: ",
            "Виберіть місто відправлення:\n1 - Саутгемптон\n2 - Шербур\n3 - Квінстаун",
            "Введіть номер міста (1, 2 або 3): ",
            "Бажаєте перевірити ще одного пасажира? (так/ні): "
        ]

    elif lang == 'fr':
        messages = [
            "Entrez la classe (1, 2 ou 3): ",
            "Entrez le sexe (1 pour l'homme, 0 pour la femme): ",
            "Entrez l'âge: ",
            "Entrez le nombre d'amis à bord: ",
            "Entrez le nombre de membres de la famille: ",
            "Choisissez la ville d'embarquement:\n1 - Southampton\n2 - Cherbourg\n3 - Queenstown",
            "Entrez le numéro de la ville (1, 2 ou 3): ",
            "Voulez-vous vérifier un autre passager? (oui/non): "
        ]

    else:
        messages = [
            "Enter cabin class (1, 2, or 3): ",
            "Enter gender (1 for male, 0 for female): ",
            "Enter age: ",
            "Enter number of friends on board: ",
            "Enter number of relatives on board: ",
            "Select departure city:\n1 - Southampton\n2 - Cherbourg\n3 - Queenstown",
            "Enter city number (1, 2, or 3): ",
            "Do you want to check another passenger? (yes/no): "
        ]

    if lang == 'ru':
        error_messages = [
            "Неправильный ввод. Класс каюты должен быть 1, 2 или 3.",
            "Неправильный ввод. Пол должен быть 1 для мужского или 0 для женского.",
            "Неправильный ввод. Возраст должен быть положительным числом.",
            "Неправильный ввод. Количество друзей на борту должно быть неотрицательным числом.",
            "Неправильный ввод. Количество родственников на борту должно быть неотрицательным числом.",
            "Неправильный ввод. Номер города должен быть 1, 2 или 3."
        ]

    elif lang == 'ua':
        error_messages = [
            "Неправильне введення. Клас каюти повинен бути 1, 2 або 3.",
            "Неправильне введення. Стать повинна бути 1 для чоловіків або 0 для жінок.",
            "Неправильне введення. Вік повинен бути додатним числом.",
            "Неправильне введення. Кількість друзів на борту повинна бути невід'ємним числом.",
            "Неправильне введення. Кількість родичів на борту повинна бути невід'ємним числом.",
            "Неправильне введення. Номер міста повинен бути 1, 2 або 3."
        ]

    elif lang == 'fr':
        error_messages = [
            "Cette réponse n'existe pas. La classe doit être 1, 2 ou 3.",
            "Cette réponse n'existe pas. Le sexe doit être 1 pour l'homme ou 0 pour la femme.",
            "Cette réponse n'existe pas. L'âge doit être un nombre positif.",
            "Cette réponse n'existe pas. Le nombre d'amis à bord doit être un nombre positif.",
            "Cette réponse n'existe pas. Le nombre de membres de la famille à bord doit être un nombre positif.",
            "Cette réponse n'existe pas. Le numéro de la ville doit être 1, 2 ou 3."
        ]

    else:
        error_messages = [
            "Invalid input. Cabin class must be 1, 2, or 3.",
            "Invalid input. Gender must be 1 for male or 0 for female.",
            "Invalid input. Age must be a positive number.",
            "Invalid input. Number of friends on board must be a non-negative number.",
            "Invalid input. Number of relatives on board must be a non-negative number.",
            "Invalid input. City number must be 1, 2, or 3."
        ]

    while True:

        while True:
            try:
                pclass = int(input(messages[0]).strip())
                if pclass not in [1, 2, 3]:
                    raise ValueError
                break
            except ValueError:
                print(f"{red}{error_messages[0]}\n")
                continue

        while True:
            try:
                sex = int(input(messages[1]).strip())
                if sex not in [0, 1]:
                    raise ValueError
                break
            except ValueError:
                print(f"{red}{error_messages[1]}\n")
                continue

        while True:
            try:
                age = int(input(messages[2]).strip())
                if age < 0 or age > 120:
                    raise ValueError
                break
            except ValueError:
                print(f"{red}{error_messages[2]}\n")
                continue

        while True:
            try:
                sibsp = int(input(messages[3]).strip())
                if sibsp < 0 or sibsp > 20:
                    raise ValueError
                break
            except ValueError:
                print(f"{red}{error_messages[3]}\n")
                continue

        while True:
            try:
                parch = int(input(messages[4]).strip())
                if parch < 0 or parch > 20:
                    raise ValueError
                break
            except ValueError:
                print(f"{red}{error_messages[4]}\n")
                continue

        print(messages[5])
        try:
            city_num = input(messages[6]).strip()
            city_num = int(city_num) if city_num else 1
            if city_num not in [1, 2, 3]:
                raise ValueError
        except ValueError:
            print(f"{red}{error_messages[5]}")
            continue

        embarked_s, embarked_c, embarked_q = get_embarked_vars(city_num)

        result = predict_survival(
            pclass, sex, age, sibsp, parch,
            embarked_c, embarked_q, embarked_s)
        print_person_info(
            pclass, sex, age, sibsp, parch,
            embarked_c, embarked_q, result, lang)

        cont = input(f"{cyan}{messages[7]}{reset}").strip().lower()
        if cont not in ['да', 'yes', 'oui', 'так']:
            break
        print("")


if __name__ == "__main__":
    print("\n\n\n\n\n\n\n\n")

    lang_choice = input(
        f"{yellow}ru - Русский, ua - Українська, fr - Français, en -"
        f" English : {reset}").strip()

    if (
        lang_choice == 'ru' or lang_choice == 'ua'
        or lang_choice == 'fr' or lang_choice == 'en'
    ):
        lang = lang_choice
        print("")

    else:
        while True:
            lang_choice = input(
                f"{yellow}ru - Русский, ua - Українська, fr - Français, en - "
                f"English : {reset}").strip()

            if (
                lang_choice == 'ru' or lang_choice == 'ua' or
                lang_choice == 'fr' or lang_choice == 'en'
            ):
                lang = lang_choice
                break

            print("")

    if lang == 'ru':
        print(
            f"\n{cyan}Программа для предсказания выживания (или нет XD) "
            "пассажиров Титаника.\nВыжил бы ты на титанике? - заполни "
            "анкету и узнай с точностью до 80%!\n")
    elif lang == 'ua':
        print(
            f"\n{cyan}Програма для передбачення виживання (чи ні XD) пасажирів"
            " Титаніка.\nЧи вижив би ти на Титаніку? - заповни анкету і"
            " дізнайся з точністю до 80%!\n")
    elif lang == 'fr':
        print(
            f"\n{cyan}Un programme pour prédire la survie (ou non XD) des"
            " passagers du Titanic.\nAurais-tu survécu sur le Titanic? - finis"
            " ce questionnaire et tu auras la réponse avec une précision "
            "jusqu'à 80%!\n")
    else:
        print(
            f"\n{cyan}A program to predict the survival (or not XD) of Titanic"
            " passengers.\nWould you have survived on the Titanic? - fill out"
            " the questionnaire and find out with up to 80% accuracy!\n")

    get_user_input()
