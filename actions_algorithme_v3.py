import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Listes des tickers et noms des actions
LISTE = ['FR0013412269', 'FR0013412020', 'FR0013412285', 'LU1834986900', 'LU1834987890', 'LU1834988781', 'LU1834985845']
LISTE_NOM = ['PANX', 'PAEEM', 'PE500', 'HLT', 'IND', 'TRV', 'FOO']

# Fonction pour obtenir le prix de l'action
def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period='1d')
    return todays_data['Close'][0]

# Calcul du vecteur d
d = [get_stock_price(ticker) for ticker in LISTE]

# Fonction pour l'optimisation du portefeuille
def optimiser_portefeuille(A, y):
    try:
        p = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        n = len(d)
        minimum_investment = 80

        prob = pulp.LpProblem("Optimisation_Portefeuille", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x{i}", lowBound=0, cat='Integer') for i in range(n)]
        z = [pulp.LpVariable(f"z{i}", lowBound=0) for i in range(n)]
        y_binary = [pulp.LpVariable(f"y{i}", cat='Binary') for i in range(n)]
        T = pulp.LpVariable("T", lowBound=sum(y))

        prob += A - pulp.lpSum([d[i] * x[i] for i in range(n)]), "Money_Not_Invested"
        prob += pulp.lpSum([d[i] * x[i] for i in range(n)]) <= A, "Budget_Constraint"
        prob += T == sum(y) + pulp.lpSum([d[i] * x[i] for i in range(n)]), "Total_Investment"

        for i in range(n):
            prob += z[i] >= (y[i] + d[i] * x[i]) - p[i] * T, f"Proportion_Upper_Constraint_{i}"
            prob += z[i] >= p[i] * T - (y[i] + d[i] * x[i]), f"Proportion_Lower_Constraint_{i}"

        for i in range(n):
            prob += d[i] * x[i] >= minimum_investment * y_binary[i], f"Minimum_Investment_Constraint_{i}"
            prob += x[i] <= y_binary[i] * (A // d[i]), f"Activation_Constraint_{i}"

        prob.solve()

        result = f"Statut: {pulp.LpStatus[prob.status]}\n\n"
        for i in range(n):
            action_name = LISTE_NOM[i]
            result += f"Nombre d'actions de type {action_name} à acheter: {x[i].varValue}\n"
           
        total_invested = sum([y[i] + d[i] * x[i].varValue for i in range(n)])
        current_distribution = [(y[i] + d[i] * x[i].varValue) / total_invested for i in range(n)]

        # Calcul de la répartition précédente
        previous_total = sum(y)
        previous_distribution = [y[i] / previous_total for i in range(n)]

        result += f"\nInvestissement total après achat : {total_invested}\n"
        result += "\nNouvelle répartition du portefeuille :\n"
        for i in range(n):
            result += f"Proportion de l'action {LISTE_NOM[i]}: {current_distribution[i] * 100:.2f}%\n"

        ecart = sum(abs(current_distribution[i] - p[i]) for i in range(n))
        result += f"\nEcart : {ecart}\n"
        result += f"\nArgent non investi: {A - pulp.lpSum([d[i] * x[i] for i in range(n)]).value()}"

        return result, current_distribution, previous_distribution, p, x, y_binary

    except Exception as e:
        return f"Erreur: {str(e)}", None, None, None, None, None

# Fonction pour créer le graphique de répartition
def plot_distribution(current_distribution, previous_distribution, target_distribution, names):
    fig, ax = plt.subplots()
    width = 0.35  # Largeur des barres
    x = range(len(names))
    
    # Barres pour la répartition précédente
    ax.bar([i - width for i in x], previous_distribution, width, color='gray', alpha=0.6, label='Répartition précédente')
    
    # Barres pour la répartition actuelle
    ax.bar([i for i in x], current_distribution, width, color='blue', alpha=0.6, label='Répartition après investissement')
    
    # Traçage de la répartition cible
    ax.plot(x, target_distribution, 'k-', linewidth=2, label='Répartition cible')  # Trait noir plus épais

    ax.set_xlabel('Action')
    ax.set_ylabel('Proportion')
    ax.set_title('Répartition du Portefeuille')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    st.pyplot(fig)

# Fonction pour convertir DataFrame en CSV
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

# Interface utilisateur avec Streamlit
st.title("Optimisation de Portefeuille")

# Entrée du montant à investir
A = st.number_input("Montant à investir", min_value=0.0, step=0.01)

# Entrée des montants pour chaque action
y = []
for i, nom in enumerate(LISTE_NOM):
    montant = st.number_input(f"Renseignez votre montant pour l'action '{nom}'", min_value=0.0, step=0.01, key=i)
    y.append(montant)

# Calculer les résultats lorsque l'utilisateur clique sur le bouton
if st.button("Calculer"):
    if A <= 0:
        st.error("Le montant à investir doit être positif.")
    else:
        result, current_distribution, previous_distribution, target_distribution, x, y_binary = optimiser_portefeuille(A, y)
        
        st.text(result)
        
        if current_distribution:
            plot_distribution(current_distribution, previous_distribution, target_distribution, LISTE_NOM)
            
            # Préparer les résultats pour téléchargement
            df_results = pd.DataFrame({
                'Type d\'action': LISTE_NOM,
                'Nombre à acheter': [x[i].varValue for i in range(len(LISTE_NOM))],
                'Acheter': ['Oui' if y_binary[i].varValue == 1 else 'Non' for i in range(len(LISTE_NOM))]
            })
            
            csv = convert_df_to_csv(df_results)
            st.download_button(
                label="Télécharger les résultats",
                data=csv,
                file_name='resultats_portefeuille.csv',
                mime='text/csv'
            )
