# Simple script to generate a bitcoin wallet
from bitcoinlib.wallets import Wallet
from bitcoinlib.mnemonic import Mnemonic
import json

def gen_wallet(wallet_name):
    # Create an instance of Mnemonic
    mnemonic_instance = Mnemonic()
    
    # Generate mnemonic
    mnemonic = mnemonic_instance.generate()

    wallet = Wallet.create(wallet_name, witness_type='segwit', keys=mnemonic,
                           network='testnet')
    # wallet details
    wallet_details = {
        'wallet_name': wallet_name,
        'mnemonic': mnemonic,
        'address': wallet.get_key().address
    }

    # Save the wallet details to a json file
    with open(f"{wallet_name}_wallet.json", 'w') as json_file:
        json.dump(wallet_details, json_file, indent=4)

    print(f"Wallet details saved to {wallet_name}_wallet.json")
    print(wallet_details)


gen_wallet("test_wallet2")
