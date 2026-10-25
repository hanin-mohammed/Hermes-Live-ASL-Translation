# Running HERMES

HERMES was developed with Python 3.11 on macOS.

## 1. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Create the MySQL database

Make sure MySQL is running, then execute:

```bash
mysql -u root -p < database/setup_hermes_database.sql
```

## 3. Set the database credentials

```bash
export HERMES_DB_HOST="localhost"
export HERMES_DB_USER="root"
export HERMES_DB_PASSWORD="your_mysql_password"
export HERMES_DB_NAME="hermes_db"
```

These values apply to the current Terminal session. Do not place a real password in a committed file.

## 4. Start the application

```bash
python hermes_app.py
```

Allow camera access if macOS requests it. Press **Start Camera**, show one of the supported signs and press **Stop Camera** before closing the app.

## Training the model again

To collect labelled hand-landmark examples:

```bash
python scripts/collect_landmark_data.py
```

To train and save a new three-neighbour KNN model:

```bash
python scripts/train_landmark_model.py
```

