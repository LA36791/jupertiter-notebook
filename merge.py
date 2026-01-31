"""
merge.py
Step-by-step dataset merger for the Hackathon final dataset.
Follows the user's specification:
  Step 1: Load CSV Data (orders)
  Step 2: Load JSON Data (users)
  Step 3: Load SQL Data (restaurants)
  Step 4: Merge the Data (left joins)
  Step 5: Create final_food_delivery_dataset.csv (orders + users + restaurants)

Usage: python merge.py
"""

import re
import csv
import sys
from pathlib import Path
import pandas as pd


def load_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Orders CSV not found at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Orders file {path} is empty")
    # normalize columns commonly expected
    df.columns = [c.strip() for c in df.columns]
    return df


def load_users(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Users JSON not found at {path}")
    try:
        df = pd.read_json(path)
    except ValueError:
        # fallback to json lines or load then json_normalize
        df = pd.read_json(path, lines=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_sql_inserts(sql_text: str, table_name: str = 'restaurants') -> pd.DataFrame:
    """Parse simple INSERT INTO `table` VALUES (...) lines into a DataFrame.
    This is tolerant and intended for the restaurants INSERTs used in this project.
    It will attempt to infer column count from the first matching INSERT.
    """
    pattern = re.compile(r"INSERT\s+INTO\s+`?" + re.escape(table_name) + r"`?\s+VALUES\s*(.+);", re.IGNORECASE | re.DOTALL)
    items = []
    for m in pattern.finditer(sql_text):
        vals = m.group(1).strip()
        # vals may be like (1,'Name','Cuisine',4.5),(2,'Name2','Cuisine2',4.2)
        # remove surrounding whitespace
        # split on '),(' but be careful with parentheses inside strings - assume simple values
        # normalize by removing leading/trailing parentheses
        if vals.startswith('(') and vals.endswith(')'):
            # split top-level records by '),(' pattern
            records = re.split(r"\),\s*\(", vals[1:-1])
        else:
            records = [vals]
        for rec in records:
            # split by commas not inside quotes (robust splitter)
            def split_sql_values(s):
                res = []
                cur = []
                in_quote = False
                quote_char = ''
                i = 0
                while i < len(s):
                    ch = s[i]
                    if ch in ("'", '"'):
                        if not in_quote:
                            in_quote = True
                            quote_char = ch
                            # don't include quote in value
                        elif ch == quote_char:
                            # handle escaped quotes ('' or "")
                            if i+1 < len(s) and s[i+1] == quote_char:
                                cur.append(quote_char)
                                i += 1
                            else:
                                in_quote = False
                                quote_char = ''
                            # don't include closing quote
                        else:
                            cur.append(ch)
                    elif ch == ',' and not in_quote:
                        res.append(''.join(cur).strip())
                        cur = []
                    else:
                        cur.append(ch)
                    i += 1
                res.append(''.join(cur).strip())
                return res

            parts = split_sql_values(rec)
            # strip surrounding quotes and whitespace
            clean = [p.strip().strip("'\"") for p in parts]
            items.append(clean)
    if not items:
        return pd.DataFrame()
    # Build dataframe with generic column names if not available
    max_cols = max(len(r) for r in items)
    cols = [f'col_{i+1}' for i in range(max_cols)]
    df = pd.DataFrame(items, columns=cols)
    return df


def load_restaurants(sql_path: Path, csv_fallback: Path = None) -> pd.DataFrame:
    # prefer parsed CSV if exists
    if csv_fallback and csv_fallback.exists():
        df = pd.read_csv(csv_fallback)
        df.columns = [c.strip() for c in df.columns]
        return df
    if not sql_path.exists():
        raise FileNotFoundError(f"Restaurants SQL not found at {sql_path}")
    text = sql_path.read_text(encoding='utf-8')
    df = parse_sql_inserts(text, table_name='restaurants')
    if df.empty:
        raise ValueError("No INSERT INTO restaurants statements found in SQL file")
    # make reasonable column names based on common schema: id,name,cuisine,rating
    # try to infer: if 4 columns -> restaurant_id,restaurant_name,cuisine,rating
    if df.shape[1] >= 4:
        df = df.rename(columns={df.columns[0]:'restaurant_id', df.columns[1]:'restaurant_name', df.columns[2]:'cuisine', df.columns[3]:'rating'})
    return df


def coerce_and_clean(final: pd.DataFrame) -> pd.DataFrame:
    # types
    if 'quantity' in final.columns:
        final['quantity'] = pd.to_numeric(final['quantity'], errors='coerce').fillna(0).astype(int)
    if 'price' in final.columns:
        final['price'] = pd.to_numeric(final['price'], errors='coerce')
    # recompute total_amount to avoid stale/misaligned values
    if all(c in final.columns for c in ('quantity','price')):
        final['total_amount'] = (final['quantity'] * final['price']).round(2)
    else:
        final['total_amount'] = pd.to_numeric(final.get('total_amount', 0), errors='coerce').fillna(0).round(2)
    # parse date if present
    if 'date' in final.columns:
        final['date'] = pd.to_datetime(final['date'], errors='coerce')
    return final


def create_final_dataset(orders: pd.DataFrame, users: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    # ensure key columns exist
    orders = orders.copy()
    users = users.copy()
    restaurants = restaurants.copy()

    # cast id columns to comparable types
    for df, col in ((orders,'user_id'), (users,'user_id')):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for df, col in ((orders,'restaurant_id'), (restaurants,'restaurant_id')):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Left join orders -> users
    merged = orders.merge(users.add_prefix('user_'), how='left', left_on='user_id', right_on='user_user_id')

    # Left join restaurants
    merged = merged.merge(restaurants.add_prefix('restaurant_'), how='left', left_on='restaurant_id', right_on='restaurant_restaurant_id')

    # flatten some names for easier analysis
    # prefer unprefixed names: e.g., user_name -> name (from users), restaurant_name etc.
    if 'user_name' in merged.columns:
        merged['name'] = merged['user_name']
    if 'user_city' in merged.columns:
        merged['city'] = merged['user_city']
    if 'user_membership' in merged.columns:
        merged['membership'] = merged['user_membership']

    # bring restaurant columns to top-level names
    if 'restaurant_name' in merged.columns:
        merged['restaurant_name'] = merged['restaurant_name']
    if 'restaurant_cuisine' in merged.columns:
        merged['cuisine'] = merged['restaurant_cuisine']
    if 'restaurant_rating' in merged.columns:
        merged['rating'] = pd.to_numeric(merged['restaurant_rating'], errors='coerce')

    # keep only necessary columns: order details + user info + restaurant info
    keep = []
    # order details (common names)
    for col in ('order_id','customer','user_id','restaurant_id','item','quantity','price','total_amount','date'):
        if col in merged.columns:
            keep.append(col)
    # user info
    for col in ('name','city','membership','user_user_id'):
        if col in merged.columns and col not in keep:
            keep.append(col)
    # restaurant info
    for col in ('restaurant_id','restaurant_name','cuisine','rating'):
        if col in merged.columns and col not in keep:
            keep.append(col)
    # source
    if '__source_file' in merged.columns:
        keep.append('__source_file')

    # if any important columns missing, still return with what we have
    final = merged[keep].copy()

    # clean and coerce
    final = coerce_and_clean(final)

    return final


def main():
    base = Path('.')
    # support common filename variants
    orders_candidates = ['order.csv', 'orders.csv']
    users_candidates = ['user.json', 'users.json']
    restaurants_sql_candidates = ['restaurrent.sql', 'restaurent.sql', 'restaurants.sql']
    restaurants_csv = base / 'restaurants.csv'

    orders_path = next((base / p for p in orders_candidates if (base / p).exists()), None)
    users_path = next((base / p for p in users_candidates if (base / p).exists()), None)
    restaurants_sql = next((base / p for p in restaurants_sql_candidates if (base / p).exists()), None)

    if orders_path is None:
        raise FileNotFoundError(f"Orders CSV not found. Tried: {orders_candidates}")
    if users_path is None:
        raise FileNotFoundError(f"Users JSON not found. Tried: {users_candidates}")

    orders = load_orders(orders_path)
    users = load_users(users_path)
    # try to use parsed CSV if present, else SQL
    restaurants = None
    if restaurants_csv.exists():
        restaurants = pd.read_csv(restaurants_csv)
    elif restaurants_sql is not None:
        restaurants = load_restaurants(restaurants_sql)
    else:
        # no restaurants source found - create empty df with expected cols
        restaurants = pd.DataFrame(columns=['restaurant_id','restaurant_name','cuisine','rating'])

    final = create_final_dataset(orders, users, restaurants)

    out = base / 'final_food_delivery_dataset.csv'
    final.to_csv(out, index=False)
    print(f'Wrote final dataset to {out} ({final.shape[0]} rows, {final.shape[1]} columns)')


if __name__ == '__main__':
    main()
