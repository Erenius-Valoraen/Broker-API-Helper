from SmartApi import SmartConnect 
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
    def __init__(self):
        self.authToken = None
        self.feedToken = None
        self.exchanges = None
        
        # Define directory paths
        self.LOG_DIR = "logs/util"
        self.JSON_DIR = "jsonLookup"

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
            authToken = data['data']['jwtToken']
            refreshToken = data['data']['refreshToken']
            feedToken = self.smartApi.getfeedToken()
            res = self.smartApi.getProfile(refreshToken)
            self.smartApi.generateToken(refreshToken)
            res=res['data']['exchanges']

            self.authToken = authToken
            self.feedToken = feedToken
            self.exchanges = res

    def prepare_resources(self, ignore_run_check=False):
        # --- [NEW] Directory Safety Checks ---
        # This checks if folders exist; if not, it creates them.
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR, exist_ok=True)
            print(f"Created missing directory: {self.LOG_DIR}")

        if not os.path.exists(self.JSON_DIR):
            os.makedirs(self.JSON_DIR, exist_ok=True)
            print(f"Created missing directory: {self.JSON_DIR}")
        # -------------------------------------

        self.RUN_LOG = os.path.join(self.LOG_DIR, "last_run.txt")
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
                print("Warning: Could not read run log. Proceeding with download.")

        print(f"[{self.today}] Status: First run of the day. Starting download and processing.")

        self.download_scrip_master()
        self.make_equity_json()
        self.make_nifty_json()

    def download_scrip_master(self):
        """
        Downloads the Scrip Master JSON file into the 'jsonLookup' folder.
        """
        URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        FILE_NAME = "ScripMaster.json"
        
        # Uses the directory ensured in prepare_resources
        target_path = os.path.join(self.JSON_DIR, FILE_NAME)

        try:
            # pd.read_json handles the HTTP request and converts the data to a DataFrame
            df = pd.read_json(URL)
            
            # Save the DataFrame back to a JSON file
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
        input_file_path = os.path.join(self.JSON_DIR, "ScripMaster.json")
        output_file_path = os.path.join(self.JSON_DIR, "nifty_options.json")

        try:
            with open(input_file_path, "r") as infile:
                data = json.load(infile)

            filtered_entries = []
            for entry in data:
                if (entry.get("name") == "NIFTY" and
                    entry.get("exch_seg") == "NFO" and
                    entry.get("instrumenttype") == "OPTIDX"):
                    filtered_entries.append(entry)

            with open(output_file_path, "w") as outfile:
                json.dump(filtered_entries, outfile, indent=4)
            
            print(f"Filtered entries saved to '{output_file_path}'")

        except FileNotFoundError:
            print(f"Error: The input file '{input_file_path}' was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_nifty_token(self, expiry, strike, option_type):
        try:
            strike_json = strike * 100
            option_type_json = option_type.upper()
            
            path = os.path.join(self.JSON_DIR, "nifty_options.json")
            
            with open(path, "r") as file:
                data = json.load(file)

            for contract in data:
                if (
                    contract.get("expiry") == expiry
                    and contract.get("strike") == strike_json
                    and option_type_json in contract.get("symbol", "")
                ):
                    return contract.get("token")

        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")

        return None
    
    def opt_ltp(self, expiry, strike, option_type):
        token = self.get_nifty_token(expiry, strike, option_type)
        if token:
            ltp_data = self.smartApi.ltpData(exchange="NFO", tradingsymbol= f"NIFTY{expiry}{strike}{option_type}", symboltoken=token)
            return ltp_data['data']['ltp']
        return 0
    
    def batch_opt_ltp(self, contracts, mode="LTP"):
        tokens = []
        for contract in contracts:
            token = str(self.get_nifty_token(contract['expiry'], contract['strike'], contract['option_type']))
            if token and token != "None":
                tokens.append(token)
        
        if not tokens:
            return {}

        data = self.smartApi.getMarketData(mode=mode, exchangeTokens= {
            "NFO": tokens
        })

        if not data.get('data'):
             return {}

        items = data["data"]["fetched"]
        ltp_map = {}
        for item in items:
            symbol = item["tradingSymbol"]
            ltp = item["ltp"]
            ltp_map[symbol] = ltp
        
        return ltp_map

    def opt_depth(self, expiry, strike, option_type):
        token = self.get_nifty_token(expiry, strike, option_type)
        if not token: return None

        data = self.smartApi.getMarketData(mode="FULL", exchangeTokens= {
            "NFO": [str(token)]
        })
        return data['data']['fetched'][0]['depth']
    
    def opt_full(self, expiry, strike, option_type):
        token = self.get_nifty_token(expiry, strike, option_type)
        if not token: return None

        data = self.smartApi.getMarketData(mode="FULL", exchangeTokens= {
            "NFO": [str(token)]
        })
        return data['data']['fetched'][0]

    def nifty_spot(self):
        return self.smartApi.ltpData(exchange="NSE", tradingsymbol="NIFTY 50", symboltoken="99926000")['data']['ltp']
    # </Options> --------------------------------------

    # <Equity> ----------------------------------------
    def make_equity_json(self):
        input_file_path = os.path.join(self.JSON_DIR, "ScripMaster.json")
        output_file_path = os.path.join(self.JSON_DIR, "equity_nse.json")

        try:
            with open(input_file_path, "r") as infile:
                data = json.load(infile)

            filtered_entries = []
            for entry in data:
                if (entry.get("exch_seg") == "NSE" and
                    entry.get("instrumenttype") == ""):
                    filtered_entries.append(entry)

            with open(output_file_path, "w") as outfile:
                json.dump(filtered_entries, outfile, indent=4)
            
            print(f"Filtered entries saved to '{output_file_path}'")

        except FileNotFoundError:
            print(f"Error: The input file '{input_file_path}' was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_equity_token(self, symbol):
        try:
            path = os.path.join(self.JSON_DIR, "equity_nse.json")
            with open(path, "r") as file:
                data = json.load(file)

            for equity in data:
                if equity.get("name") == symbol.strip().upper():
                    return equity.get("token")

        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
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
    
    def eq_candles(self, symbol, interval, from_date, to_date):
        token = self.get_equity_token(symbol)
        data = self.smartApi.getCandleData(
            historicDataParams= {
            "exchange":"NSE",
            "symboltoken":str(token),
            "interval":interval,
            "fromdate":from_date,
            "todate":to_date}
        )
        return data['data']
    # </Equity> --------------------------------------

# ------------------------------------------------------------------
# CLASSES: Contract, OptionChain, Expiry
# (These remain mostly the same, but OptionChain now creates directory if passed)
# ------------------------------------------------------------------

class Contract:
    def __init__(self, chain, contract_data):
        self.chain = chain
        self.data = contract_data
        self.token = contract_data.get('token')
        self.symbol = contract_data.get('symbol')
        self.expiry = contract_data.get('expiry')
        
        raw_strike = contract_data.get('strike', 0)
        self.strike = float(raw_strike) / 100
        
        if "PE" in self.symbol[-2:]:
            self.option_type = "PE"
        else:
            self.option_type = "CE"
    
    def ltp(self):
        return self.chain.api.opt_ltp(self.expiry, self.strike, self.option_type)

    def depth(self):
        return self.chain.api.opt_depth(self.expiry, self.strike, self.option_type)

    def bid(self):
        d = self.depth()
        return d.get('buy', [{}])[0].get('price', None) if d else None
    
    def ask(self):
        d = self.depth()
        return d.get('sell', [{}])[0].get('price', None) if d else None

    def full(self):
        return self.chain.api.opt_full(self.expiry, self.strike, self.option_type)

    def __repr__(self):
        return f"<Contract: {self.symbol} | Strike: {self.strike}>"

class OptionChain:
    def __init__(self, api, json_path='jsonLookup/nifty_options.json'):
        self.api = api
        self.json_path = json_path
        self.chain_data = []
        
        # Ensure directory exists for safety
        folder = os.path.dirname(self.json_path)
        if folder and not os.path.exists(folder):
             os.makedirs(folder, exist_ok=True)

        self.spot_price = float(self.api.nifty_spot())
        self.load_chain()

    def load_chain(self):
        try:
            if not os.path.exists(self.json_path):
                print(f"Warning: {self.json_path} not found. Chain is empty. Run api.prepare_resources() first.")
                return

            with open(self.json_path, 'r') as f:
                self.chain_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")

    def find(self, offset, expiry_date, option_type):
        candidates = []
        
        for item in self.chain_data:
            item_expiry = item.get('expiry')
            item_symbol = item.get('symbol')
            
            is_ce = "CE" in item_symbol or item_symbol.endswith("CE")
            current_type = "CE" if is_ce else "PE"

            if item_expiry == expiry_date and current_type == option_type:
                candidates.append(Contract(self, item))

        if not candidates:
            print(f"No contracts found for {expiry_date} {option_type}")
            return None

        candidates.sort(key=lambda x: x.strike)
        atm_contract = min(candidates, key=lambda x: abs(x.strike - self.spot_price))
        atm_index = candidates.index(atm_contract)
        
        target_index = None
        if offset > 100:
            target_strike = float(offset)
            for idx, contract in enumerate(candidates):
                if contract.strike == target_strike:
                    return contract
            print(f"No contract found with strike {target_strike}")
            return None
        else:
            target_index = atm_index + offset

        if 0 <= target_index < len(candidates):
            return candidates[target_index]
        else:
            print(f"Strike offset {offset} is out of bounds for this chain.")
            return None
        
    def display(self, expiry_date, num_strikes=10):
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

        sorted_strikes = sorted(chain_map.keys())
        atm_strike = min(sorted_strikes, key=lambda x: abs(x - self.spot_price))
        atm_index = sorted_strikes.index(atm_strike)

        start_idx = max(0, atm_index - num_strikes)
        end_idx = min(len(sorted_strikes), atm_index + num_strikes + 1)
        visible_strikes = sorted_strikes[start_idx:end_idx]

        contracts = []
        for strike in visible_strikes:
            contracts.append({'expiry': expiry_date, 'strike': strike, 'option_type': 'CE'})
            contracts.append({'expiry': expiry_date, 'strike': strike, 'option_type': 'PE'})
        
        ltp_map = self.api.batch_opt_ltp(contracts)

        table_rows = []
        for strike in visible_strikes:
            ce_contract = chain_map[strike]['CE']
            pe_contract = chain_map[strike]['PE']

            # Safe get for prices
            ce_key = f"NIFTY{expiry_date.replace('20', '')}{int(strike)}CE"
            pe_key = f"NIFTY{expiry_date.replace('20', '')}{int(strike)}PE"

            try:
                ce_price = f"{ltp_map.get(ce_key, 0.0):.2f}" if ce_contract else "-"
                pe_price = f"{ltp_map.get(pe_key, 0.0):.2f}" if pe_contract else "-"
            except:
                ce_price, pe_price = "-", "-"
            
            strike_display = f"{strike:.2f}"
            is_atm = (strike == atm_strike)
            
            if is_atm:
                fmt_ce = f"{Back.CYAN}{Fore.BLACK} {ce_price} {Style.RESET_ALL}"
                fmt_st = f"{Back.CYAN}{Fore.BLACK} {strike_display} {Style.RESET_ALL}"
                fmt_pe = f"{Back.CYAN}{Fore.BLACK} {pe_price} {Style.RESET_ALL}"
                strike_display = f"--> {strike_display} <--"
            else:
                fmt_ce = f"{Fore.GREEN}{ce_price}{Style.RESET_ALL}"
                fmt_st = f"{Style.BRIGHT}{strike_display}{Style.RESET_ALL}"
                fmt_pe = f"{Fore.RED}{pe_price}{Style.RESET_ALL}"

            table_rows.append([fmt_ce, fmt_st, fmt_pe])

        headers = [f"{Fore.GREEN}CALLS (LTP){Style.RESET_ALL}", "STRIKE", f"{Fore.RED}PUTS (LTP){Style.RESET_ALL}"]
        print(f"\nOption Chain for {Fore.YELLOW}{expiry_date}{Style.RESET_ALL} (Spot: {self.spot_price})")
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid", stralign="center"))

class Expiry:
    def __init__(self, json_path="jsonLookup/nifty_options.json"):
        # Folder check
        folder = os.path.dirname(json_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        if not os.path.exists(json_path):
             print(f"Warning: {json_path} not found. Expiry functions will fail until setup is run.")
             self.expiries = []
             return

        with open(json_path, "r") as f:
            data = json.load(f)

        self.expiries = sorted(
            {self._parse_date(item["expiry"]) for item in data}
        )

    def _parse_date(self, expiry_str):
        return datetime.datetime.strptime(expiry_str, "%d%b%Y").date()

    def _format_date(self, dt):
        return dt.strftime("%d%b%Y").upper()

    def _find_nearest_expiry(self, target_date):
        if not self.expiries: return None
        
        pos = bisect_left(self.expiries, target_date)
        if pos == 0: return self.expiries[0]
        if pos == len(self.expiries): return self.expiries[-1]

        before = self.expiries[pos - 1]
        after = self.expiries[pos]

        if abs((after - target_date).days) < abs((target_date - before).days):
            return after
        else:
            return before

    def W(self, from_date=None):
        if from_date is None: from_date = datetime.date.today()
        target = from_date + datetime.timedelta(weeks=1)
        res = self._find_nearest_expiry(target)
        return self._format_date(res) if res else None

    def W2(self, from_date=None):
        if from_date is None: from_date = datetime.date.today()
        target = from_date + datetime.timedelta(weeks=2)
        res = self._find_nearest_expiry(target)
        return self._format_date(res) if res else None

    def M(self, from_date=None):
        if from_date is None: from_date = datetime.date.today()
        month = from_date.month + 1
        year = from_date.year
        if month > 12:
            month -= 12
            year += 1
        day = min(from_date.day, 28)
        target = datetime.date(year, month, day)
        res = self._find_nearest_expiry(target)
        return self._format_date(res) if res else None

    def M6(self, from_date=None):
        if from_date is None: from_date = datetime.date.today()
        month = from_date.month + 6
        year = from_date.year
        if month > 12:
            month -= 12
            year += 1
        day = min(from_date.day, 28)
        target = datetime.date(year, month, day)
        res = self._find_nearest_expiry(target)
        return self._format_date(res) if res else None

    def Y(self, from_date=None):
        if from_date is None: from_date = datetime.date.today()
        target = datetime.date(from_date.year + 1, from_date.month, min(from_date.day, 28))
        res = self._find_nearest_expiry(target)
        return self._format_date(res) if res else None