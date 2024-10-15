from bitcoinlib.wallets import Wallet
from bitcoinlib.transactions import Transaction
import requests
import json

def sendback():
    address = 'tb1qerzrlxcfu24davlur5sqmgzzgsal6wusda40er'
    msg = "Send back coins when done: tb1qerzrlxcfu24davlur5sqmgzzgsal6wusda40er"
    return print(msg)

# Print the contents of a .md file
def printMD(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            print(content)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_wallet_info(wallet_name):
    wallet = Wallet(wallet_name)
    keys = wallet.get_key().address
    return keys

# Obtain balance and address from the wallet
def get_testnet_balance_from_address(address):
    # Blockstream testnet explorer API URL
    url = f"https://blockstream.info/testnet/api/address/{address}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        balance_satoshis = data['chain_stats']['funded_txo_sum']
        balance_btc = balance_satoshis / 1e8 # Convert satoshis to BTC
        print(f"Address: {address}")
        print(f"Balance for address: {balance_btc:.8f} tBTC")
        return balance_btc
    else:
        print(f"Error fetching balance: {response.status_code}")


# Fetch transactions for the admin wallet
def fetch_transactions(address):
    url = f'https://blockstream.info/testnet/api/address/{address}/txs'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Save the full transaction details to a .json file
        with open(f"{address}_transactions.json", 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Transaction details saved to {address}_transactions.json")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

wallet_name = "test_wallet2"
get_testnet_balance_from_address(get_wallet_info(wallet_name))
sendback()
