#
# ['gitcommit', 'gitcommitdate', 'hostname', 'ndim', 'kernel_type', 'interp_type', 'tree_height', 'interp_order', 'error']
# Je souhaite comparer le champ de la colonne 'error' avec le champ 10.0^(1-o) pour o dans la colonne 'interp_order' lorsque la valeur de  'interp_order' est < 12
#     je souhaite afficher toutes les lignes telsque si comparaison est False alors j'extrais le tuple associé sux colonnes 'ndim', 'kernel_type', 'interp_type', 'tree_height' et pour ce tuple de valeur j'extraie les lignes
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file="scalfmm_accuracy.csv"

df = pd.read_csv(file)
coeff = 3.0
# Calcul de 10^(1 - interp_order)
df['computed_value'] = coeff*(10.0 ** (1 - df['interp_order']))
xref=  df['interp_order'].unique()
yref= 10.0 ** (1 - xref)

# Comparaison: 'error' < 'computed_value' seulement lorsque 'interp_order' < 12
# et 'interp_order' n'est pas 0 ni 4
df['comparison'] = np.where(
    (df['interp_order'] < 12) , #& (~df['interp_type'].isin([0, 4])),  # Comparaison si interp_order < 12 et pas 0 ou 4
    df['error'] < df['computed_value'],  # Comparaison réelle
    True  # Si interp_order >= 12 ou 0/4, la comparaison est True
)
# Filtrer les lignes où la comparaison est fausse
df_faux_comparaison = df[~df['comparison']]

# 2. Extraire les tuples uniques ('ndim', 'kernel_type', 'interp_type', 'tree_height') pour lesquels la comparaison est fausse
tuples_faux = df_faux_comparaison[['ndim', 'kernel_type', 'interp_type', 'tree_height']].drop_duplicates()
# 3. Créer un DataFrame vide pour stocker toutes les lignes correspondant à ces tuples
df_resultat = pd.DataFrame(columns=df.columns)
for _, couple in tuples_faux.iterrows():
    ndim, kernel, interp, tree = couple['ndim'], couple['kernel_type'], couple['interp_type'], couple['tree_height']

    # Filtrer les lignes du DataFrame d'origine qui ont ce tuple
    result = df[(df['ndim'] == ndim) & (df['kernel_type'] == kernel) &
                (df['interp_type'] == interp) & (df['tree_height'] == tree)]

    # Afficher les lignes pour chaque tuple
    print(f"Résultats pour ndim = {ndim}, kernel_type = {kernel}, interp_type = {interp}, tree_height = {tree}:")
    # print(result)
    print("\n" + "="*50 + "\n")
    # Ajouter les lignes filtrées au DataFrame résultat
    df_resultat = pd.concat([df_resultat, result])

print(df_resultat.head(30))

# Réinitialiser l'index du DataFrame résultat
df_resultat.reset_index(drop=True, inplace=True)
df_resultat.head()
for d in range(1,4):
    plt.figure()

    for (ndim, kernel_type, interp_type, tree_height), group in df_resultat.groupby(['ndim', 'kernel_type', 'interp_type', 'tree_height']):
        if ndim == d:
            x = group['interp_order']
            y = group['error']
            plt.plot(x, y, label=f"k={kernel_type}-i={interp_type}-h={tree_height}")
            plt.yscale('log')

    plt.plot(xref,yref,'-x')
    plt.xlabel('Order')
    plt.ylabel('Relative norm-2 error')
    plt.title('Error vs. Order')
    plt.legend()
    plt.savefig(f"scalfmm_wrong_accuracy_d={d}.pdf", bbox_inches='tight')
