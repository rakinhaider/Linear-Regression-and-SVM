import pandas as pd
import numpy as np
import preprocess as pr
import utils as util
import split as splt

def get_one_hot_encoding(data, column, value):
    df = data[data[column] == value]
    df = df.drop(columns=[column])
    related_columns = [col for col in df.columns if col.startswith(column+'__')]
    df = df[related_columns]
    df = df.drop_duplicates()
    assert(len(df) <= 1)
    if len(df) == 0:
        return [0 for col in df[column].unique().tolist()[:-1]]
    return df.values.tolist()[0]

if __name__ == "__main__":

    if util.final == True:
        columns, data = util.readFile('dating-full.csv',None)
    else:
        columns, data = util.readFile('test_dataset.csv')

    # Answer to question 1.i
    data = data[:6500]

    # Preprocess similar to assignment 2 1.i
    pr.stripQuotes(data, ['race', 'race_o', 'field'])

    # Preprocess similar to assignment 2 1.ii
    pr.toLowerCase(data, ['field'])

    # Preprocess similar to assignment 2 1.iv
    pr.normalizeColumns(data, util.psParticipants, util.psPartners)


    # Answer to question 1.ii
    dum = pd.get_dummies(data, prefix=[val+'_' for val in util.categorical], columns=util.categorical)

    # Rearranging columns
    indices = [list(data.columns).index(col) for col in util.categorical]
    for i in range(len(indices)):
        dum[util.categorical[i]] = data[util.categorical[i]]
        cols = dum.columns.tolist()
        cols = cols[:indices[i]] + [util.categorical[i]] + cols[indices[i]:-1]
        dum = dum[cols]

    decision_index = dum.columns.tolist().index('decision')
    cols = dum.columns.tolist()
    cols = cols[:decision_index] + cols[decision_index+1:] +['decision']

    data = dum[cols]

    # Drop last columns. So the one-hot-encoding of last value will be an all zero vector.
    for col in util.categorical:
        column_values = data[col].sort_values(ascending=False).values
        data = data.drop(columns=[col+'__'+column_values[0]])

    values = ['female', 'Black/African American', 'Other', 'economics']

    for i in range(len(util.categorical)):
        one_hot_encoding = get_one_hot_encoding(data, util.categorical[i], values[i])
        print('Mapped vector for {} in column {}: {}'.format(values[i], util.categorical[i], one_hot_encoding))
        # print(len(one_hot_encoding))


    data = data.drop(columns=util.categorical)

    # Answer to question 1.iii
    train, test = splt.split(data, random_state=25, frac=0.2)
    splt.save_train_and_test_split(train, test)