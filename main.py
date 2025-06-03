import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Chargement et préparation des données + modèle
df = pd.read_csv(resource_path("titanic.csv"))
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
df["Embarked"].fillna("S", inplace=True)
median_ages = df.groupby("Pclass")["Age"].median()


def imputer_age(row):
    if pd.isnull(row["Age"]):
        return median_ages[row["Pclass"]]
    return row["Age"]


df["Age"] = df.apply(imputer_age, axis=1)
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
X = df.drop("Survived", axis=1)
y = df["Survived"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_scaled, y)
features = list(X.columns)

embarked_map = {
    1: (0, 0),  # Southampton
    2: (1, 0),  # Cherbourg
    3: (0, 1),  # Queenstown
}


def predire_survie(pclass, sexe, age, sibsp, parch, embarked_c, embarked_q):
    moyennes_tarifs = {1: 84, 2: 20, 3: 13}
    tarif = moyennes_tarifs.get(pclass, 0)
    data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sexe],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [tarif],
        "Embarked_C": [embarked_c],
        "Embarked_Q": [embarked_q],
    })
    data = data[features]
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction[0] == 1


class TitanicApp(App):
    def build(self):
        Window.size = (500, 600)
        self.title = "Prédiction de survie Titanic"

        main_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Champs de saisie et sélection
        # Classe
        main_layout.add_widget(Label(text="Classe (1, 2 ou 3) :", size_hint_y=None, height=30))
        self.pclass_input = Spinner(text="1", values=("1", "2", "3"), size_hint_y=None, height=40)
        main_layout.add_widget(self.pclass_input)

        # Sexe
        main_layout.add_widget(Label(text="Sexe :", size_hint_y=None, height=30))
        self.sexe_input = Spinner(text="Homme", values=("Homme", "Femme"), size_hint_y=None, height=40)
        main_layout.add_widget(self.sexe_input)

        # Age
        main_layout.add_widget(Label(text="Âge (0-120) :", size_hint_y=None, height=30))
        self.age_input = TextInput(text="30", multiline=False, input_filter="int", size_hint_y=None, height=40)
        main_layout.add_widget(self.age_input)

        # SibSp
        main_layout.add_widget(Label(text="Nombre d'amis à bord (0-20) :", size_hint_y=None, height=30))
        self.sibsp_input = TextInput(text="0", multiline=False, input_filter="int", size_hint_y=None, height=40)
        main_layout.add_widget(self.sibsp_input)

        # Parch
        main_layout.add_widget(Label(text="Nombre de membres de la famille (0-20) :", size_hint_y=None, height=30))
        self.parch_input = TextInput(text="0", multiline=False, input_filter="int", size_hint_y=None, height=40)
        main_layout.add_widget(self.parch_input)

        # Ville d'embarquement
        main_layout.add_widget(Label(text="Ville d'embarquement :", size_hint_y=None, height=30))
        self.embarked_input = Spinner(text="Southampton", values=("Southampton", "Cherbourg", "Queenstown"), size_hint_y=None, height=40)
        main_layout.add_widget(self.embarked_input)

        # Bouton prédiction
        self.pred_button = Button(text="Prédire la survie", size_hint_y=None, height=50)
        self.pred_button.bind(on_press=self.on_predict)
        main_layout.add_widget(self.pred_button)

        # Label résultat
        self.result_label = Label(text="", size_hint_y=None, height=120)
        main_layout.add_widget(self.result_label)

        # Bouton reset
        self.reset_button = Button(text="Réinitialiser", size_hint_y=None, height=40)
        self.reset_button.bind(on_press=self.on_reset)
        main_layout.add_widget(self.reset_button)

        return main_layout

    def on_predict(self, instance):
        # Récupérer et valider les données
        try:
            pclass = int(self.pclass_input.text)
            sexe = 1 if self.sexe_input.text == "Homme" else 0
            age = int(self.age_input.text)
            sibsp = int(self.sibsp_input.text)
            parch = int(self.parch_input.text)
            embarked_str = self.embarked_input.text
            if not (1 <= pclass <= 3):
                raise ValueError("La classe doit être 1, 2 ou 3.")
            if not (0 <= age <= 120):
                raise ValueError("L'âge doit être entre 0 et 120.")
            if not (0 <= sibsp <= 20):
                raise ValueError("Le nombre d'amis doit être entre 0 et 20.")
            if not (0 <= parch <= 20):
                raise ValueError("Le nombre de membres de la famille doit être entre 0 et 20.")
        except ValueError as e:
            self.result_label.text = f"Erreur de saisie : {e}"
            return

        # Map ville vers colonnes embarquées
        ville_num = {"Southampton": 1, "Cherbourg": 2, "Queenstown": 3}[embarked_str]
        embarked_c, embarked_q = embarked_map[ville_num]

        survie = predire_survie(pclass, sexe, age, sibsp, parch, embarked_c, embarked_q)

        sexe_str = "un homme" if sexe == 1 else "une femme"
        ville = embarked_str
        survie_str = "aurait survécu" if survie else "n'aurait pas survécu"

        self.result_label.text = (
            f"Ce passager était {sexe_str}, de la classe {pclass}.\n"
            f"Il/elle avait {age} ans, avec {sibsp} amis et {parch} membres de sa famille à bord.\n"
            f"Il/elle a embarqué à {ville}.\n"
            f"Selon le modèle, il/elle {survie_str}."
        )

    def on_reset(self, instance):
        self.pclass_input.text = "1"
        self.sexe_input.text = "Homme"
        self.age_input.text = "30"
        self.sibsp_input.text = "0"
        self.parch_input.text = "0"
        self.embarked_input.text = "Southampton"
        self.result_label.text = ""


if __name__ == "__main__":
    TitanicApp().run()
