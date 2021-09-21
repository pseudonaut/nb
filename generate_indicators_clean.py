from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score as sk_score
from generate_indicators_v2_helper import *

pd.set_option('display.max_columns', None)

tcoins = [
    'ALGO', 'ATOM', 'BAL', 'BAND', 'BCH', 'BTC', 'CGLD', 'COMP', 'DASH', 'EOS', 'ETC', 'ETH', 'KNC',
    'LINK', 'LTC', 'MKR', 'NMR', 'OMG', 'OXT', 'REN', 'REP', 'UMA', 'UNI', 'XLM', 'XTZ', 'YFI', 'ZRX'
]

print_out = False

factors = ['ema_1', 'ema_2', 'ema_3', 'ema_4']
# factors = ['ema_1_diff', 'ema_2_diff', 'ema_3_diff', 'ema_4_diff']

print(factors)

for coin in tcoins:

    # Load datas
    df_train = pd.read_csv('training_ohlc/' + coin + '_cbpro_ohlc_60m.csv', sep=',')
    df_test = pd.read_csv('testing_ohlc/' + coin + '_cbpro_ohlc_60m.csv', sep=',')

    # Clean NaN values
    df_train = dropna(df_train)
    df_test = dropna(df_test)

    df_train = generate_indicators(df_train)
    df_test = generate_indicators(df_test)

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    X_train = df_train.loc[:, factors]
    y_train = df_train.loc[:, 'up']
    X_test = df_test.loc[:, factors]
    y_test = df_test.loc[:, 'up']

    model = GaussianNB().fit(X_train, y_train)
    predicted_y = model.predict(X_test)
    accuracy_score = sk_score(y_test, predicted_y)

    return_data = generate_returns(df_test, predicted_y)

    if print_out:
        generate_chart(df_test, predicted_y, coin, factors)
        time.sleep(1)

    print(coin, accuracy_score)
