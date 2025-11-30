from SmartApi import SmartConnect #or from SmartApi.smartConnect import SmartConnect
import pyotp
from logzero import logger

import requests
import os
import pandas as pd
import datetime
from datetime import date
import json

import math
from tabulate import tabulate
from colorama import init, Fore, Back, Style

from bisect import bisect_left

# Initialize colorama for Windows compatibility
init(autoreset=True)

class API:
    def login(self, api_key, client_code, pin, qr_value):

        api_key = api_key
        username = client_code
        pwd = pin
        self.smartApi = SmartConnect(api_key)
        try:
            token = qr_value
            totp = pyotp.TOTP(token).now()
        except Exception as e:
            logger.error("Invalid Token: The provided token is not valid.")
            raise e

        correlation_id = "abcde"
        data = self.smartApi.generateSession(username, pwd, totp)

        if data['status'] == False:
            logger.error(data)
            
        else:
            # login api call
            # logger.info(f"You Credentials: {data}")
            authToken = data['data']['jwtToken']
            refreshToken = data['data']['refreshToken']
            # fetch the feedtoken
            feedToken = self.smartApi.getfeedToken()
            # fetch User Profile
            res = self.smartApi.getProfile(refreshToken)
            self.smartApi.generateToken(refreshToken)
            res=res['data']['exchanges']

    def prepare_resources(self, ignore_run_check=False):
        self.RUN_LOG = "logs/util/last_run.txt"

        self.today = date.today().strftime("%Y-%m-%d")
        
        # 1. Check if the function ran successfully today
        if os.path.exists(self.RUN_LOG) and not ignore_run_check:
            try:
                with open(self.RUN_LOG, 'r') as f:
                    last_run_date = f.read().strip()
                    if last_run_date == self.today:
                        print(f"[{self.today}] Status: Already run successfully today. Skipping download.")
                        return
            except IOError:
                # Handle case where file exists but is unreadable/corrupt
                print("Warning: Could not read run log. Proceeding with download.")

        print(f"[{self.today}] Status: First run of the day. Starting download and processing.")

        self.download_scrip_master()
        self.make_equity_json()
        self.make_nifty_json()

    def download_scrip_master(self):
        """
        Checks the last run date and downloads the Scrip Master JSON file 
        using pandas.read_json() only once per day into the 'jsonLookup' folder.
        """
        # --- Configuration ---
        URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        DIR = "jsonLookup"
        FILE_NAME = "ScripMaster.json"

        
        # 2. Ensure the directory exists
        target_path = os.path.join(DIR, FILE_NAME)
        os.makedirs(DIR, exist_ok=True)

        # 3. Download, read, and save the JSON using Pandas
        try:
            # pd.read_json handles the HTTP request and converts the data to a DataFrame
            df = pd.read_json(URL)
            
            # Save the DataFrame back to a JSON file (orient='records' is good for lists of objects)
            df.to_json(target_path, orient='records', indent=4)
                
            print(f"Success: File downloaded and saved to '{target_path}'.")

            # 4. Log successful run date
            with open(self.RUN_LOG, 'w') as f:
                f.write(self.today)
                
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to download the file. Check URL or network connection. Error details: {e}")
        except Exception as e:
            print(f"An error occurred during Pandas processing or file write: {e}")

    # <Options> ----------------------------------------
    def make_nifty_json(self):
        # Define input and output file paths
        input_file_path = "jsonLookup/ScripMaster.json"
        output_file_path = "jsonLookup/nifty_options.json"

        try:
            # 1. Read data from the input JSON file
            with open(input_file_path, "r") as infile:
                data = json.load(infile)

            # 2. Filter the data based on specified criteria
            filtered_entries = []
            for entry in data:
                if (entry.get("name") == "NIFTY" and
                    entry.get("exch_seg") == "NFO" and
                    entry.get("instrumenttype") == "OPTIDX"):
                    filtered_entries.append(entry)

            # 3. Ensure the output directory exists
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 4. Write the filtered entries to the new JSON file
            with open(output_file_path, "w") as outfile:
                json.dump(filtered_entries, outfile, indent=4)
            
            print(f"Filtered entries from '{input_file_path}' have been successfully saved to '{output_file_path}'")

        except FileNotFoundError:
            print(f"Error: The input file '{input_file_path}' was not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{input_file_path}'. Check if the file is valid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_nifty_token(self, expiry, strike, option_type):
        """
        Finds and returns the token number for a given option contract.

        Args:
            expiry (str): The expiry date in the format "DDMMMYYYY" (e.g., "30DEC2025").
            strike (int): The strike price (e.g., 2000000).
            option_type (str): The option type, either "CE" or "PE".

        Returns:
            str: The token number of the matching contract, or None if not found.
        """
        try:
            # The strike price in the JSON is 100 times the actual price.
            strike_json = strike * 100
            # The option type needs to be uppercase to match the symbol.
            option_type_json = option_type.upper()

            with open("jsonLookup/nifty_options.json", "r") as file:
                data = json.load(file)

            for contract in data:
                if (
                    contract.get("expiry") == expiry
                    and contract.get("strike") == strike_json
                    and option_type_json in contract.get("symbol", "")
                ):
                    return contract.get("token")

        except FileNotFoundError:
            print("Error: The file 'jsonLookup.nifty_options.json' was not found.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")

        return None
    
    def opt_ltp(self, expiry, strike, option_type):
        token = self.get_nifty_token(expiry, strike, option_type)
        ltp_data = self.smartApi.ltpData(exchange="NFO", tradingsymbol= f"NIFTY{expiry}{strike}{option_type}", symboltoken=token)
        return ltp_data['data']['ltp']
    
    def batch_opt_ltp(self, contracts, mode="LTP"):
        tokens = []
        for contract in contracts:
            token = str(self.get_nifty_token(contract['expiry'], contract['strike'], contract['option_type']))
            if token:
                tokens.append(token)
        
        data = self.smartApi.getMarketData(mode=mode, exchangeTokens= {
            "NFO": tokens
        })

        items = data["data"]["fetched"]
        ltp_map = {}
        for item in items:
            symbol = item["tradingSymbol"]
            ltp = item["ltp"]
            ltp_map[symbol] = ltp
        
        return ltp_map

    def opt_depth(self, expiry, strike, option_type):
        token = self.get_nifty_token(expiry, strike, option_type)

        data = self.smartApi.getMarketData(mode="FULL", exchangeTokens= {
            "NFO": [str(token)]
        })

        return data['data']['fetched'][0]['depth']

    def nifty_spot(self):
        return self.smartApi.ltpData(exchange="NSE", tradingsymbol="NIFTY 50", symboltoken="99926000")['data']['ltp']
    # </Options> --------------------------------------

    # <Equity> ----------------------------------------
    def make_equity_json(self):
        # Define input and output file paths
        input_file_path = "jsonLookup/ScripMaster.json"
        output_file_path = "jsonLookup/equity_nse.json"

        try:
            # 1. Read data from the input JSON file
            with open(input_file_path, "r") as infile:
                data = json.load(infile)

            # 2. Filter the data based on specified criteria
            filtered_entries = []
            for entry in data:
                if (entry.get("exch_seg") == "NSE" and
                    entry.get("instrumenttype") == ""):
                    filtered_entries.append(entry)

            # 3. Ensure the output directory exists
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 4. Write the filtered entries to the new JSON file
            with open(output_file_path, "w") as outfile:
                json.dump(filtered_entries, outfile, indent=4)
            
            print(f"Filtered entries from '{input_file_path}' have been successfully saved to '{output_file_path}'")

        except FileNotFoundError:
            print(f"Error: The input file '{input_file_path}' was not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{input_file_path}'. Check if the file is valid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_equity_token(self, symbol):
        """
        Finds and returns the token number for a given equity symbol.

        Args:
            symbol (str): The equity symbol (e.g., "RELIANCE").

        Returns:
            str: The token number of the matching equity, or None if not found.
        """
        try:
            with open("jsonLookup/equity_nse.json", "r") as file:
                data = json.load(file)

            for equity in data:
                if equity.get("name") == symbol.strip().upper():
                    return equity.get("token")

        except FileNotFoundError:
            print("Error: The file 'jsonLookup/equity_nse.json' was not found.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")

        return None

    def eq_ltp(self, symbol):
        token = self.get_equity_token(symbol)
        ltp_data = self.smartApi.ltpData(exchange="NSE", tradingsymbol=symbol.strip().upper(), symboltoken=token)
        return ltp_data['data']['ltp']
    
    def batch_eq_ltp(self, symbols, mode="LTP"):
        tokens = []
        for symbol in symbols:
            token = str(self.get_equity_token(symbol))
            if token:
                tokens.append(token)
        
        data = self.smartApi.getMarketData(mode=mode, exchangeTokens= {
            "NSE": tokens
        })

        items = data["data"]["fetched"]
        ltp_map = {}
        for item in items:
            symbol = item["tradingSymbol"]
            ltp = item["ltp"]
            ltp_map[symbol] = ltp
        
        return ltp_map

    def eq_depth(self, symbol):
        token = self.get_equity_token(symbol)

        data = self.smartApi.getMarketData(mode="FULL", exchangeTokens= {
            "NSE": [str(token)]
        })

        return data['data']['fetched'][0]['depth']
    # </Equity> --------------------------------------


class Contract:
    """
    Represents a single option contract.
    Allows calling .ltp() directly on the object.
    """
    def __init__(self, chain, contract_data):
        self.chain = chain
        self.data = contract_data
        
        self.token = contract_data.get('token')
        self.symbol = contract_data.get('symbol')
        self.expiry = contract_data.get('expiry')
        
        # Handling strike price scaling (JSON typically has 3000000 for 30000.00)
        raw_strike = contract_data.get('strike', 0)
        self.strike = float(raw_strike) / 100
        
        # Determine Option Type (CE/PE)
        if "PE" in self.symbol[-2:]:
            self.option_type = "PE"
        else:
            self.option_type = "CE"

    def ltp(self):
        """
        Calls the parent API's opt_ltp method using this contract's details.
        """
        return self.chain.api.opt_ltp(self.expiry, self.strike, self.option_type)

    def depth(self):
        return self.chain.api.opt_depth(self.expiry, self.strike, self.option_type)

    def bid(self):
        return self.depth().get('buy', [{}])[0].get('price', None)
    
    def ask(self):
        return self.depth().get('sell', [{}])[0].get('price', None)

    def __repr__(self):
        return f"<Contract: {self.symbol} | Strike: {self.strike}>"

class OptionChain:

    """
    Manages loading the option chain and finding contracts relative to ATM.
    Extends the 'api' class.
    """
    def __init__(self, api, json_path='jsonLookup/nifty_options.json'):
        # Initialize the parent API class
        self.api = api
        
        self.json_path = json_path
        self.chain_data = []
        self.spot_price = float(self.api.nifty_spot())
        self.load_chain()

    def load_chain(self):
        """Loads and parses the JSON file."""
        try:
            # Check if file exists (for safety in this demo)
            if not os.path.exists(self.json_path):
                print(f"Warning: {self.json_path} not found. Using empty list.")
                return

            with open(self.json_path, 'r') as f:
                self.chain_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")

    def find(self, offset, expiry_date, option_type):
        """
        Finds a contract based on offset from ATM.
        
        Args:
            offset (int): 0 for ATM, 1 for next strike higher, -1 for next strike lower.
            spot_price (float): Current underlying price to calculate ATM.
            expiry_date (str): Exact expiry string, e.g., "29JUN2027".
            option_type (str): "CE" or "PE".
            
        Returns:
            Contract: A Contract object with .ltp() method, or None if not found.
        """
        
        # 1. Filter relevant contracts for this expiry and type
        candidates = []
        
        for item in self.chain_data:
            # Extract basic data
            item_expiry = item.get('expiry')
            item_symbol = item.get('symbol')
            
            # Simple check to distinguish CE vs PE based on symbol suffix
            # Assumes standard NFO naming convention or relies on user input correctness
            is_ce = "CE" in item_symbol or item_symbol.endswith("CE")
            is_pe = "PE" in item_symbol or item_symbol.endswith("PE")
            
            current_type = "CE" if is_ce else "PE"

            if item_expiry == expiry_date and current_type == option_type:
                # Create a temporary wrapper to help with sorting, 
                # or just use the dictionary if performance is critical.
                # Here we create the object immediately.
                candidates.append(Contract(self, item))

        if not candidates:
            print(f"No contracts found for {expiry_date} {option_type}")
            return None

        # 2. Sort candidates by Strike Price
        candidates.sort(key=lambda x: x.strike)

        # 3. Find ATM (The strike closest to Spot Price)
        # We use min() with a key function calculating the absolute difference
        atm_contract = min(candidates, key=lambda x: abs(x.strike - self.spot_price))
        
        # Get the index of the ATM contract
        atm_index = candidates.index(atm_contract)
        
        # 4. Apply Offset
        target_index = atm_index + offset

        # Boundary checks
        if 0 <= target_index < len(candidates):
            return candidates[target_index]
        else:
            print(f"Strike offset {offset} is out of bounds for this chain.")
            return None
        
    def display(self, expiry_date, num_strikes=10):
        """
        Prints a pretty table of the option chain.
        
        Args:
            expiry_date (str): The expiry to filter by (e.g., "29JUN2027")
            spot_price (float): Used to highlight the ATM row.
            num_strikes (int): How many strikes to show above/below ATM.
        """

        # 1. Organize data into a dictionary: { strike: {'CE': contract, 'PE': contract} }
        chain_map = {}
        
        for item in self.chain_data:
            if item.get('expiry') != expiry_date:
                continue
                
            c = Contract(self, item)
            if c.strike not in chain_map:
                chain_map[c.strike] = {'CE': None, 'PE': None}
            
            chain_map[c.strike][c.option_type] = c

        if not chain_map:
            print(f"{Fore.RED}No data found for expiry: {expiry_date}{Style.RESET_ALL}")
            return

        # 2. Sort strikes
        sorted_strikes = sorted(chain_map.keys())

        # 3. Find ATM Index
        # We find the strike closest to spot_price
        atm_strike = min(sorted_strikes, key=lambda x: abs(x - self.spot_price))
        atm_index = sorted_strikes.index(atm_strike)

        # 4. Slice the list to show contracts around ATM
        start_idx = max(0, atm_index - num_strikes)
        end_idx = min(len(sorted_strikes), atm_index + num_strikes + 1)
        visible_strikes = sorted_strikes[start_idx:end_idx]

        contracts = []

        for strike in visible_strikes:
            ce_contract = {'expiry': expiry_date, 'strike': strike, 'option_type': 'CE'}
            pe_contract = {'expiry': expiry_date, 'strike': strike, 'option_type': 'PE'}
            contracts.append(ce_contract)
            contracts.append(pe_contract)
        
        ltp_map = self.api.batch_opt_ltp(contracts)

        # 5. Build Table Rows
        table_rows = []
        
        for strike in visible_strikes:
            ce_contract = chain_map[strike]['CE']
            pe_contract = chain_map[strike]['PE']

            # Get Prices
            try:
                ce_price = f"{ltp_map[f"NIFTY{expiry_date.replace("20", "")}{int(strike)}CE"]:.2f}" if ce_contract else "-"
                pe_price = f"{ltp_map[f"NIFTY{expiry_date.replace("20", "")}{int(strike)}PE"]:.2f}" if pe_contract else "-"
            except KeyError:
                ce_price = "-"
                pe_price = "-"
            
            strike_display = f"{strike:.2f}"

            # Formatting logic
            is_atm = (strike == atm_strike)
            
            if is_atm:
                # Highlight the whole row for ATM
                # We use Back.WHITE + Fore.BLACK for high contrast
                # Note: Tabulate handles color codes in strings well
                fmt_ce = f"{Back.CYAN}{Fore.BLACK} {ce_price} {Style.RESET_ALL}"
                fmt_st = f"{Back.CYAN}{Fore.BLACK} {strike_display} {Style.RESET_ALL}"
                fmt_pe = f"{Back.CYAN}{Fore.BLACK} {pe_price} {Style.RESET_ALL}"
                
                # Add an indicator arrow
                strike_display = f"--> {strike_display} <--"
            else:
                # Colorize prices: CE usually Green, PE usually Red (or vice versa depending on pref)
                fmt_ce = f"{Fore.GREEN}{ce_price}{Style.RESET_ALL}"
                fmt_st = f"{Style.BRIGHT}{strike_display}{Style.RESET_ALL}"
                fmt_pe = f"{Fore.RED}{pe_price}{Style.RESET_ALL}"
                
                # If ATM row logic applied previously, we need to ensure these are consistent
                if is_atm:
                    pass # Handled above

            table_rows.append([fmt_ce, fmt_st, fmt_pe])

        # 6. Print Table
        headers = [f"{Fore.GREEN}CALLS (LTP){Style.RESET_ALL}", "STRIKE", f"{Fore.RED}PUTS (LTP){Style.RESET_ALL}"]
        
        print(f"\nOption Chain for {Fore.YELLOW}{expiry_date}{Style.RESET_ALL} (Spot: {self.spot_price})")
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid", stralign="center"))



class Expiry:
    def __init__(self, json_path="jsonLookup/nifty_options.json"):
        # Load JSON data
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract unique expiries and sort by date
        self.expiries = sorted(
            {self._parse_date(item["expiry"]) for item in data}
        )

    # ----------------------
    # DATE PARSING UTILITIES
    # ----------------------

    def _parse_date(self, expiry_str):
        """Convert '29JUN2027' → datetime.date"""
        return datetime.datetime.strptime(expiry_str, "%d%b%Y").date()

    def _format_date(self, dt):
        """Convert datetime.date → '29JUN2027' """
        return dt.strftime("%d%b%Y").upper()

    def _find_nearest_expiry(self, target_date):
        """Return nearest available expiry date."""
        pos = bisect_left(self.expiries, target_date)

        if pos == 0:
            return self.expiries[0]
        if pos == len(self.expiries):
            return self.expiries[-1]

        before = self.expiries[pos - 1]
        after = self.expiries[pos]

        # Return whichever is closer
        if abs((after - target_date).days) < abs((target_date - before).days):
            return after
        else:
            return before

    # ----------------------
    # EXPIRY COMPUTATION
    # ----------------------

    def W(self, from_date=None):
        if from_date is None:
            from_date = datetime.date.today()
        target = from_date + datetime.timedelta(weeks=1)
        return self._format_date(self._find_nearest_expiry(target))

    def W2(self, from_date=None):
        if from_date is None:
            from_date = datetime.date.today()
        target = from_date + datetime.timedelta(weeks=2)
        return self._format_date(self._find_nearest_expiry(target))

    def M(self, from_date=None):
        if from_date is None:
            from_date = datetime.date.today()

        # Add 1 month safely
        month = from_date.month + 1
        year = from_date.year
        if month > 12:
            month -= 12
            year += 1

        day = min(from_date.day, 28)  # Prevent date overflow
        target = datetime.date(year, month, day)

        return self._format_date(self._find_nearest_expiry(target))

    def M6(self, from_date=None):
        if from_date is None:
            from_date = datetime.date.today()

        month = from_date.month + 6
        year = from_date.year
        if month > 12:
            month -= 12
            year += 1

        day = min(from_date.day, 28)
        target = datetime.date(year, month, day)

        return self._format_date(self._find_nearest_expiry(target))

    def Y(self, from_date=None):
        if from_date is None:
            from_date = datetime.date.today()

        target = datetime.date(from_date.year + 1, from_date.month, min(from_date.day, 28))
        return self._format_date(self._find_nearest_expiry(target))

expiry = Expiry()
