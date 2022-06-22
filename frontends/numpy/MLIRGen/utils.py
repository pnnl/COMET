#
# Copyright 2022 Battelle Memorial Institute
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions 
# and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

class HashTable:
  
    # Create empty bucket list of given size
    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()
        self.dims_table = self.create_buckets()
        self.indexlbls_table = self.create_buckets()
        self.gen_table = self.create_buckets()
  
    def create_buckets(self):
        return [[] for _ in range(self.size)]
  
    # Insert values into hash map
    def set_val(self, key, val):
        
        # Get the index from the key
        # using hash function
        hashed_key = hash(key) % self.size
          
        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            if record_key == key:
                found_key = True
                break

        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))


    def set_dims(self, key, val:list):
        
        hashed_key = hash(key) % self.size      
        bucket = self.dims_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            if record_key == key:
                found_key = True
                break
  
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    def set_dims_labels(self, key, val:list):
        
        hashed_key = hash(key) % self.size      
        bucket = self.indexlbls_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            if record_key == key:
                found_key = True
                break
  
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    def add_vars(self, key, ret_arrays:list):
        
        # Get the index from the key
        # using hash function
        hashed_key = hash(key) % self.size
          
        # Get the bucket corresponding to index
        bucket = self.gen_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            # check if the bucket has same key as
            # the key to be inserted
            if record_key == key:
                found_key = True
                break
  
        # If the bucket has same key as the key to be inserted,
        # Update the key value
        # Otherwise append the new key-value pair to the bucket
        if found_key:
            bucket[index] = (key, ret_arrays)
        else:
            bucket.append((key, ret_arrays))


  
    # Return searched value with specific key
    def get_val(self, key):
        
        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size
          
        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            # check if the bucket has same key as 
            # the key being searched
            if record_key == key:
                found_key = True
                break
  
        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"


    def get_dims(self, key):
        
        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size
          
        # Get the bucket corresponding to index
        bucket = self.dims_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            # check if the bucket has same key as 
            # the key being searched
            if record_key == key:
                found_key = True
                break
  
        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"

    def get_dims_labels(self, key):
        
        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size
          
        # Get the bucket corresponding to index
        bucket = self.indexlbls_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            # check if the bucket has same key as 
            # the key being searched
            if record_key == key:
                found_key = True
                break
  
        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"

    def get_vars(self, key):
        
        hashed_key = hash(key) % self.size
          
        bucket = self.gen_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            if record_key == key:
                found_key = True
                break
  
        if found_key:
            return record_val
        else:
            return "No record found"
  
    # Remove a value with specific key
    def delete_val(self, key):
      
        hashed_key = hash(key) % self.size
          
        bucket = self.hash_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

    def delete_dims(self, key):
        
        hashed_key = hash(key) % self.size
          
        bucket = self.dims_table[hashed_key]
  
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
              
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

  
    # To print the items of hash map
    def __str__(self):
        return "".join(str(item) for item in self.hash_table)


class oprs_info:

    einsum_targets = {}
    dict_input_arrays = {}
    dict_input_dim_lbls = {}
    dict_out_dim_lbls = {}
    dict_opr_types = {}
    dict_operators = {}

    def __init__(self, einsumno,target,input_arrays,input_dim_lbls,output_dim_lbls,opr_type:str,op:str):
        self.einsum_targets[einsumno] = target
        self.dict_input_arrays[einsumno] = input_arrays
        self.dict_input_dim_lbls[einsumno] = input_dim_lbls
        self.dict_out_dim_lbls[einsumno] = output_dim_lbls
        self.dict_opr_types[einsumno] = opr_type
        self.dict_operators[einsumno] = op
        pass

    @classmethod
    def get_einsum_target(self,key):
        return self.einsum_targets[key]

    @classmethod
    def get_input_arrays(self,key):
        return self.dict_input_arrays[key]

    @classmethod
    def get_input_dims_lbls(self,key):
        return self.dict_input_dim_lbls[key]

    @classmethod
    def get_out_dims_lbls(self,key):
        return self.dict_out_dim_lbls[key]

    @classmethod
    def get_opr_type(self,key):
        return self.dict_opr_types[key]

    @classmethod
    def get_op(self,key):
        return self.dict_operators[key]

        
