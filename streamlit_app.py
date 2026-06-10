import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_excel("Football_Matches_2425_2526.xlsx")


# дата
df["Дата"] = pd.to_datetime(dict(
    year=df["Год"],
    month=df["Месяц"],
    day=df["День"]
))

df = df.sort_values("Дата")
home = df[["Дата", "Команда-1", "Команда-2", "Счёт"]].copy()
home[["g1", "g2"]] = home["Счёт"].str.split("-", expand=True).astype(int)

home = home.rename(columns={"Команда-1": "team", "g1": "goals_for", "g2": "goals_against"})
home = home[["Дата", "team", "goals_for", "goals_against"]]

away = df[["Дата", "Команда-2", "Команда-1", "Счёт"]].copy()
away[["g1", "g2"]] = away["Счёт"].str.split("-", expand=True).astype(int)

away = away.rename(columns={"Команда-2": "team", "g2": "goals_for", "g1": "goals_against"})
away = away[["Дата", "team", "goals_for", "goals_against"]]

long_df = pd.concat([home, away]).sort_values(["team", "Дата"])
long_df["form_goals_for"] = (
    long_df.groupby("team")["goals_for"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

long_df["form_goals_against"] = (
    long_df.groupby("team")["goals_against"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
long_df["target_next_goals"] = (
    long_df.groupby("team")["goals_for"]
    .shift(-1)
)
model_df = long_df.dropna().copy()

features = ["form_goals_for", "form_goals_against"]
X = model_df[features]
y = model_df["target_next_goals"]


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)










def predict_match(home_team, away_team, date, long_df, model):
    
    date = pd.to_datetime(date)

    # берем только матчи ДО этой даты (важно!)
    history = long_df[long_df["Дата"] < date]

    # последние 5 матчей формы home
    home_hist = history[history["team"] == home_team].tail(5)
    home_for = home_hist["goals_for"].mean()
    home_against = home_hist["goals_against"].mean()

    # последние 5 матчей формы away
    away_hist = history[history["team"] == away_team].tail(5)
    away_for = away_hist["goals_for"].mean()
    away_against = away_hist["goals_against"].mean()

    # если мало данных
    home_for = 0 if pd.isna(home_for) else home_for
    home_against = 0 if pd.isna(home_against) else home_against
    away_for = 0 if pd.isna(away_for) else away_for
    away_against = 0 if pd.isna(away_against) else away_against

    # X для модели (две команды → два ряда)
    X_match = pd.DataFrame([
        [home_for, home_against],
        [away_for, away_against]
    ], columns=["form_goals_for", "form_goals_against"])

    preds = model.predict(X_match)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals_pred": preds[0],
        "away_goals_pred": preds[1],
    }



teams = ['Burnley', 'Almeria', 'Sevilla', 'Brighton', 'Ath Bilbao',
       'Arsenal', 'Las Palmas', 'Bournemouth', 'Sociedad', 'Newcastle',
       'Sheffield United', 'Everton', 'Villarreal', 'Brentford', 'Getafe',
       'Chelsea', 'Celta', 'Man United', 'Ath Madrid', 'Cadiz',
       "Nott'm Forest", 'Valencia', 'Mallorca', 'Osasuna', 'Inter',
       'Genoa', 'Frosinone', 'Empoli', 'Fulham', 'Liverpool', 'Wolves',
       'Tottenham', 'Man City', 'Roma', 'Sassuolo', 'Lecce', 'Udinese',
       'Betis', 'Girona', 'Aston Villa', 'West Ham', 'Barcelona',
       'Torino', 'Granada', 'Alaves', 'Crystal Palace', 'Bologna',
       'Verona', 'Milan', 'Monza', 'Juventus', 'Lazio', 'Napoli',
       'Fiorentina', 'Salernitana', 'Cagliari', 'Vallecano', 'Luton',
       'Real Madrid', 'Atalanta', 'Ipswich', 'Parma', 'Valladolid',
       'Leicester', 'Espanol', 'Southampton', 'Leganes', 'Venezia',
       'Como', 'Sunderland', 'Leeds', 'Elche', 'Levante', 'Oviedo',
       'Cremonese', 'Pisa']

team1 = st.selectbox(
    "Команда 1",
    teams,
    key="team1"
)

team2 = st.selectbox(
    "Команда 2",
    teams,
    key="team2"
)

date = st.date_input("Дата матча")

if st.button("Предсказать"):

    a = predict_match(
        home_team=team1,
        away_team=team2,
        date=date,
        long_df=long_df,
        model=model
    )

def round_to_half(x):
    whole = int(x)
    decimal = x - whole
    
    if decimal < 0.5:
        return whole
    else:
        return whole + 0.5

total = round_to_half(a['home_goals_pred']) + round_to_half(a['away_goals_pred'])

st.write(f"{team1}: {a['home_goals_pred']:.2f}")
st.write(f"{team2}: {a['away_goals_pred']:.2f}")
st.write(f"Тотал: {total:.2f}")
