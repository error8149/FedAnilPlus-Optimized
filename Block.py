# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
import copy
import json
from hashlib import sha256
class Block:
	def __init__(self, idx, previous_block_hash=None, transactions=None, nonce=0, miner_rsa_pub_key=None, mined_by=None, mining_rewards=None, pow_proof=None, signature=None):
		self._idx = idx
		self._previous_block_hash = previous_block_hash
		self._transactions = transactions
		self._nonce = nonce
		# miner specific
		self._miner_rsa_pub_key = miner_rsa_pub_key
		self._mined_by = mined_by
		self._mining_rewards = mining_rewards
		# FedAnil+: Cache transaction hash for memory-efficient resyncing
		import pickle
		from hashlib import sha256
		if transactions is not None:
			self._transactions_hash = sha256(pickle.dumps(transactions)).hexdigest()
		else:
			self._transactions_hash = sha256(pickle.dumps([])).hexdigest()
		
		# validator specific
		# the hash of the current block, calculated by compute_hash
		self._pow_proof = pow_proof
		self._signature = signature

	# compute_hash() also used to return value for block verification
	def compute_hash(self, hash_entire_block=False):
		# Optimized for 12GB RAM: Use cached transaction hash instead of raw data
		if hash_entire_block:
			# Use a version of dict that excludes the bulky raw transactions
			block_content = {k: v for k, v in self.__dict__.items() if k != '_transactions'}
		else:
			# For PoW/Verification, we only need the fixed fields + stored hash of transactions
			block_content = {k: v for k, v in self.__dict__.items() if k not in ['_pow_proof', '_signature', '_mining_rewards', '_transactions']}
		
		# Add the pre-calculated transaction hash (constant even after transactions are freed)
		block_content['_transactions_hash'] = self._transactions_hash
		
		# Sort items for deterministic hashing
		from hashlib import sha256
		return sha256(str(sorted(block_content.items())).encode('utf-8')).hexdigest()

	def return_block_dict_for_signature(self):
		return {k: v for k, v in self.__dict__.items() if k != '_signature'}

	def remove_signature_for_verification(self):
		self._signature = None

	def set_pow_proof(self, the_hash):
		self._pow_proof = the_hash

	def nonce_increment(self):
		self._nonce += 1

	# returners of the private attributes
	
	def return_previous_block_hash(self):
		return self._previous_block_hash

	def return_block_idx(self):
		return self._idx
	
	def return_pow_proof(self):
		return self._pow_proof
	
	def return_miner_rsa_pub_key(self):
		return self._miner_rsa_pub_key

	''' Miner Specific '''
	def set_previous_block_hash(self, hash_to_set):
		self._previous_block_hash = hash_to_set

	def add_verified_transaction(self, transaction):
		# after verified in cross_verification()
		# transactions can be both local_enterprises' or validators' transactions
		self._transactions.append(transaction)
		# Update cached hash after adding a transaction
		import pickle
		from hashlib import sha256
		self._transactions_hash = sha256(pickle.dumps(self._transactions)).hexdigest()

	def set_nonce(self, nonce):
		self._nonce = nonce

	def set_mined_by(self, mined_by):
		self._mined_by = mined_by
	
	def return_mined_by(self):
		return self._mined_by

	def set_signature(self, signature):
		# signed by mined_by node
		self._signature = signature

	def return_signature(self):
		return self._signature

	def set_mining_rewards(self, mining_rewards):
		self._mining_rewards = mining_rewards

	def return_mining_rewards(self):
		return self._mining_rewards
	
	def free_tx(self):
		# Optimized for 12GB RAM: Clear model weights but keep metadata for rewards/blacklisting
		try:
			if hasattr(self, '_transactions') and self._transactions:
				# 1. Clear individual local update params (very heavy)
				for tx_list_key in ['valid_validator_sig_transacitons', 'invalid_validator_sig_transacitons']:
					if tx_list_key in self._transactions:
						for tx in self._transactions[tx_list_key]:
							if 'local_updates_params' in tx:
								# Clear the heavy weights but keep the transaction metadata
								tx['local_updates_params'] = None
				
				# 2. We keep 'global_update_params' if we might need to "jump" to this model state
				# but we could also clear it if Enterprise.py is modified to cache it elsewhere.
				# For now, let's keep it as it's just one model per block, not dozens.
		except Exception as e:
			print(f"Error in free_tx: {e}")

	def return_transactions(self):
		if not hasattr(self, '_transactions'):
			# Fallback to prevent AttributeErrors if something went wrong
			return {'valid_validator_sig_transacitons': [], 'invalid_validator_sig_transacitons': [], 'global_update_params': None}
		return self._transactions