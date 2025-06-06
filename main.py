# TitanicApp avec probabilité de survie, historique, passager aléatoire et mode sombre/clair

import pandas as pd
import sys
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.progressbar import ProgressBar


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
    "Southampton": (0, 0),
    "Cherbourg": (1, 0),
    "Queenstown": (0, 1),
}


def predire_survie_proba(pclass, sexe, age, sibsp, parch, embarked_c, embarked_q):
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
    proba = model.predict_proba(data_scaled)[0][1]
    return proba


class TitanicApp(App):
    def build(self):
        Window.size = (500, 700)
        self.title = "Prédiction de survie Titanic"
        self.historique = []
        self.dark_mode = False

        self.main_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Mode sombre / clair
        self.mode_toggle = ToggleButton(text="Mode sombre : OFF", size_hint_y=None, height=40)
        self.mode_toggle.bind(on_press=self.toggle_mode)
        self.main_layout.add_widget(self.mode_toggle)

        # Champs
        self.pclass_input = Spinner(text="1", values=("1", "2", "3"), size_hint_y=None, height=40)
        self.sexe_input = Spinner(text="Homme", values=("Homme", "Femme"), size_hint_y=None, height=40)
        self.age_input = TextInput(text="30", multiline=False, input_filter="int", size_hint_y=None, height=40)
        self.sibsp_input = TextInput(text="0", multiline=False, input_filter="int", size_hint_y=None, height=40)
        self.parch_input = TextInput(text="0", multiline=False, input_filter="int", size_hint_y=None, height=40)
        self.embarked_input = Spinner(text="Southampton", values=("Southampton", "Cherbourg", "Queenstown"), size_hint_y=None, height=40)

        for label, widget in [
            ("Classe (1, 2 ou 3) :", self.pclass_input),
            ("Sexe :", self.sexe_input),
            ("Âge (0-120) :", self.age_input),
            ("Nombre d'amis à bord (0-20) :", self.sibsp_input),
            ("Nombre de membres de la famille (0-20) :", self.parch_input),
            ("Ville d'embarquement :", self.embarked_input),
        ]:
            self.main_layout.add_widget(Label(text=label, size_hint_y=None, height=30))
            self.main_layout.add_widget(widget)

        # Boutons
        bouton_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        self.pred_button = Button(text="Prédire la survie")
        self.pred_button.bind(on_press=self.on_predict)
        bouton_layout.add_widget(self.pred_button)

        self.alea_button = Button(text="Passager aléatoire")
        self.alea_button.bind(on_press=self.on_random)
        bouton_layout.add_widget(self.alea_button)

        self.main_layout.add_widget(bouton_layout)

        self.result_label = Label(text="", size_hint_y=None, height=120)
        self.main_layout.add_widget(self.result_label)

        self.progress_bar = ProgressBar(max=100, value=0, size_hint_y=None, height=20)
        self.main_layout.add_widget(self.progress_bar)

        # Historique
        self.histo_label = Label(text="Historique des prédictions :", size_hint_y=None, height=30)
        self.main_layout.add_widget(self.histo_label)
        self.histo_box = BoxLayout(orientation="vertical", size_hint_y=None)
        self.histo_box.bind(minimum_height=self.histo_box.setter('height'))
        scroll = ScrollView(size_hint=(1, None), size=(500, 150))
        scroll.add_widget(self.histo_box)
        self.main_layout.add_widget(scroll)

        return self.main_layout

    def on_predict(self, instance):
        try:
            pclass = int(self.pclass_input.text)
            sexe = 1 if self.sexe_input.text == "Homme" else 0
            age = int(self.age_input.text)
            sibsp = int(self.sibsp_input.text)
            parch = int(self.parch_input.text)
            embarked_c, embarked_q = embarked_map[self.embarked_input.text]

            if not (0 <= age <= 120 and 0 <= sibsp <= 20 and 0 <= parch <= 20):
                raise ValueError("Valeurs hors limites.")
        except Exception as e:
            self.result_label.text = f"Erreur : {e}"
            return

        proba = predire_survie_proba(pclass, sexe, age, sibsp, parch, embarked_c, embarked_q)
        survie_str = "aurait survécu" if proba >= 0.5 else "n'aurait pas survécu"
        sexe_str = "un homme" if sexe == 1 else "une femme"
        ville = self.embarked_input.text

        texte = (
            f"Ce passager était {sexe_str}, classe {pclass}, âgé(e) de {age} ans.\n"
            f"Il/elle avait {sibsp} amis et {parch} membres de famille à bord.\n"
            f"Embarqué(e) à {ville}.\n"
            f"Probabilité de survie : {int(proba*100)}% \u2192 Il/elle {survie_str}."
        )

        self.result_label.text = texte
        self.progress_bar.value = int(proba * 100)

        self.historique.append(texte)
        self.histo_box.add_widget(Label(text=texte, size_hint_y=None, height=100))

    def on_random(self, instance):
        self.pclass_input.text = str(random.choice([1, 2, 3]))
        self.sexe_input.text = random.choice(["Homme", "Femme"])
        self.age_input.text = str(random.randint(1, 80))
        self.sibsp_input.text = str(random.randint(0, 5))
        self.parch_input.text = str(random.randint(0, 5))
        self.embarked_input.text = random.choice(["Southampton", "Cherbourg", "Queenstown"])

    def toggle_mode(self, instance):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            Window.clearcolor = (0.1, 0.1, 0.1, 1)
            instance.text = "Mode sombre : ON"
        else:
            Window.clearcolor = (1, 1, 1, 1)
            instance.text = "Mode sombre : OFF"


if __name__ == "__main__":
    TitanicApp().run()
