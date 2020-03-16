import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale


def process_stock_data(path, years=["2014"]):

    x_data_list = []
    y_data_list = []
    for year in years:
        file = path + "/" + year + "_Financial_Data.csv"

        data = pd.read_csv(file, delimiter=",")
        data = data.dropna(thresh=10)
        data = data.fillna(method='bfill')
        data_np = data.to_numpy()
        x_data = data_np[:, 1:-3]

        x_data_list.append(x_data)
        y_data_list.append(data_np[:, -1])

    x_data = np.vstack(x_data_list)
    y_data = np.hstack(y_data_list)

    # attempt to scale the data for standardization
    # x_data = np.divide((x_data - x_data.min(0)), x_data.ptp(0), where=x_data.ptp(0)!=0)
    # x_data = x_data.astype(np.float)
    # x_data = np.nan_to_num(x_data)
    x_data = scale(x_data)
    x_data = np.nan_to_num(x_data)

    # return every column except the name and the class
    return x_data, y_data.astype(int)


def process_census_data(path):
    file = path + "/adult.data"
    census_data = np.ndarray([32561, 15])
    row_counter = 0

    workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
                 'Never-worked', '?']
    education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                 '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?']
    marital = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
               'Married-AF-spouse', '?']
    occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                  'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                  'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?']
    relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?']
    race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?']
    sex = ['Female', 'Male', '?']
    native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                      'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                      'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                      'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                      'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']
    classification = ['<=50K', '>50K']

    with open(file, 'r') as f:
        lines = f.readlines()
        for row in lines:
            col_data = row.split(',')
            for col in range(0, len(col_data)):
                if col == 1:
                    census_data[row_counter, col] = workclass.index(col_data[col].strip(' '))
                elif col == 3:
                    census_data[row_counter, col] = education.index(col_data[col].strip(' '))
                elif col == 5:
                    census_data[row_counter, col] = marital.index(col_data[col].strip(' '))
                elif col == 6:
                    census_data[row_counter, col] = occupation.index(col_data[col].strip(' '))
                elif col == 7:
                    census_data[row_counter, col] = relationship.index(col_data[col].strip(' '))
                elif col == 8:
                    census_data[row_counter, col] = race.index(col_data[col].strip(' '))
                elif col == 9:
                    census_data[row_counter, col] = sex.index(col_data[col].strip(' '))
                elif col == 13:
                    census_data[row_counter, col] = native_country.index(col_data[col].strip(' '))
                elif col == 14:
                    census_data[row_counter, col] = classification.index(col_data[col].strip())
                else:
                    census_data[row_counter, col] = col_data[col]
            row_counter += 1
    # normalize the continuous rows
    census_data[:, 2] = scale(census_data[:, 2])
    census_data[:, 2] = np.nan_to_num(census_data[:, 2])

    census_data[:, 10] = scale(census_data[:, 10])
    census_data[:, 10] = np.nan_to_num(census_data[:, 10])

    census_data[:, 11] = scale(census_data[:, 11])
    census_data[:, 11] = np.nan_to_num(census_data[:, 11])

    census_data[:, 12] = scale(census_data[:, 12])
    census_data[:, 12] = np.nan_to_num(census_data[:, 12])

    # ret_data = panda_df.to_numpy()
    return census_data[:, :-2], census_data[:, -1]


# Custom process the gas data to put in the form [y x1 x2 ... xn]
# in an np array
# return [X] y
def process_gas_data(path):
    gas_data = np.ndarray([13910, 129])
    row_counter = 0
    for batch in range(1, 11):
        file = path + "/batch" + str(batch) + ".dat"
        with open(file, 'r') as f:
            lines = f.readlines()
            for row in lines:
                col_data = row.split()
                gas_data[row_counter, 0] = int(col_data[0])
                for col in range(1, 129):
                    gas_data[row_counter, col] = float(col_data[col].split(':')[1])
                row_counter += 1

    x_data = scale(gas_data[:, 1:])
    x_data = np.nan_to_num(x_data)

    return x_data, gas_data[:, 0]


# Custom process the pen data to put in the form [y x1 x2 ... xn]
# in an np array
# return [X] y
def process_pen_data(path):
    pen_data = np.ndarray([7494, 17])
    row_counter = 0
    file = path + "/pendigits.tra"
    with open(file, 'r') as f:
        lines = f.readlines()
        for row in lines:
            col_data = row.split(',')
            pen_data[row_counter, 0] = int(col_data[16])
            for col in range(0, 16):
                pen_data[row_counter, col + 1] = int(col_data[col])
            row_counter += 1

    return pen_data[:, 1:], pen_data[:, 0]


def process_spam_data(path, noise=0.0):
    full_path = path + "/spam_encoded.csv"
    data = pd.read_csv(full_path, delimiter=",")

    data_np = data.to_numpy()

    # add noise if desired
    for row in data_np:
        if np.random.random() < noise:
            row[0] = np.random.randint(0, 1)

    return data_np[:, 2:-1], data_np[:, 0].astype(int)


# Custom process the bank data to put in the form [y x1 x2 ... xn]
# in an np array
# return [X] y
def process_bank_data(path):
    full_path = path + "/bank-full.csv"

    data = pd.read_csv(full_path, delimiter=";")

    job_class = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
                 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    marital_class = ['divorced', 'married', 'single', 'unknown']
    edu_class = ['primary', 'secondary', 'tertiary', 'unknown']
    yes_no_class = ['no', 'yes']
    contact_class = ['unknown', 'cellular', 'telephone']
    month_class = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    poutcome_class = ['unknown', 'success', 'failure', 'other']

    # convert labels to integers
    data['job'] = data['job'].apply(job_class.index)
    data['marital'] = data['marital'].apply(marital_class.index)
    data['education'] = data['education'].apply(edu_class.index)
    data['default'] = data['default'].apply(yes_no_class.index)
    data['housing'] = data['housing'].apply(yes_no_class.index)
    data['loan'] = data['loan'].apply(yes_no_class.index)
    data['contact'] = data['contact'].apply(contact_class.index)
    data['month'] = data['month'].apply(month_class.index)
    data['poutcome'] = data['poutcome'].apply(poutcome_class.index)
    data['y'] = data['y'].apply(yes_no_class.index)

    data_bank = data.to_numpy()
    return data_bank[:, 0:-1], data_bank[:, -1]


def process_letter_data(path):
    full_path = path + "/letter-recognition.data"
    data = np.ndarray([20000, 17])
    row_counter = 0

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
               'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    with open(full_path, 'r') as f:
        lines = f.readlines()
        for row in lines:
            col_data = row.split(',')
            data[row_counter, 0] = letters.index(col_data[0].lower())
            for col in range(1, 17):
                data[row_counter, col] = int(col_data[col])
            row_counter += 1

    return data[:, 1:], data[:, 0]

def main():
    stock_data = process_stock_data('./../Datasets/Stocks', ["2016", "2017"])


if __name__ == '__main__':
    main()