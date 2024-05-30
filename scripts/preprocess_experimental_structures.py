from tqdm.auto import tqdm
import os

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.internal_coords import *
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBIO import PDBIO


cif_parser = MMCIFParser(QUIET=True)
pdb_parser = PDBParser(QUIET=True)
io = PDBIO()
# default 1.4. 1.8 is less than a single amino acid chain break and accounts for a freak bond 
# (pdb outliers and disordered AF regions)
IC_Chain.MaxPeptideBond = 1.8 


def processPdb(pdb, chain, out_dir, process, cif_path, use_cif, chain_in_filename=False):
    '''
    Extracts the specified chain from cif file. Cif is used instead of pdb because for large structures pdb might be absent.
    If "process" is True, will replace modified residues to standard ones, delete heterogens, add missing atoms (not residues).
    The chain in the resulting file will be renamed to "A".
    cif_path is a folder with cif files <pdb>.cif (e.g. 1hho.cif) for input.
    '''
    # set output file name
    out_file = f"{pdb}_{chain}.pdb"
    if out_dir != None:
        out_file = os.path.join(out_dir, out_file)  
        
    # extract chain and save to new file
    if chain_in_filename:
        cif_file = os.path.join(cif_path, f"{pdb}_{chain}.cif")
        pdb_file = os.path.join(cif_path, f"{pdb}_{chain}.pdb")
    else:
        cif_file = os.path.join(cif_path, pdb+'.cif')
        pdb_file = os.path.join(cif_path, pdb+'.pdb')

    if use_cif:
        structure = cif_parser.get_structure(" ", cif_file)
    else: # parse pdb file instead of cif
        structure = pdb_parser.get_structure(" ", pdb_file)
    
    # replace chain if it not present in the structure
    chain_ids = [cur_chain.get_id() for cur_chain in structure[0]]
    if chain not in chain_ids:
        print(f'Replaced chain {chain} to {chain_ids[0]} for structure {pdb}')
        chain = chain_ids[0] # select the first one, can be done better

    structure = structure[0][chain] # if NMR, will extract model 0
    # parent should be detached, otherwise won't let to rename chain to A if it was originally present in the structure
    structure.detach_parent() 
    # rename chain to A, because if chain is not a single letter, io.save will fail
    structure.id = 'A' 
    io.set_structure(structure)
    io.save(out_file)
        
    # processing: replaces modified residues to standard ones, deletes heterogens, adds missing atoms (not residues)
    print('pdbfixer process', process)
    if process == True:
        print(f"pdbfixer {out_file} --keep-heterogens=none --replace-nonstandard --add-atoms=heavy --output={out_file}")
        res = os.system(f"pdbfixer {out_file} --keep-heterogens=none --replace-nonstandard --add-atoms=heavy --output={out_file}")
        print(res)


if __name__ == '__main__':
    # path with all pdb structures
    initial_structure_path = '/workspace/data/docking/test_structures/PDB/'  # '<path to folder with protein structures>'

    # path where preprocessed pdb structures will be stored
    preprocessed_structure_path = '/workspace/data/docking/test_structures/PDB/tmp_preprocessed_structures/' #'<path to where to save preprocessed structures>'

    os.makedirs(preprocessed_structure_path, exist_ok=False)
    all_ids = [fname for fname in sorted(os.listdir(initial_structure_path)) 
               if fname.endswith('.pdb') or fname.endswith('.cif')]

    for fname in tqdm(all_ids):
        pdb_path = os.path.join(initial_structure_path, fname)
        full_pdb_id = '.'.join(fname.split('.')[:-1])

        # we suppose that fname contains some ID and chain separated by '_'
        splitted = full_pdb_id.split('_')
        if len(splitted) == 1:
            pdb_id = full_pdb_id
            chain = 'A'
            chain_in_filename = False
        else:
            pdb_id = '_'.join(splitted[:-1])
            chain = splitted[-1]
            chain_in_filename = True
        print(pdb_id, chain, fname)
        processPdb(pdb_id, chain, preprocessed_structure_path, process=True,
                cif_path=initial_structure_path, use_cif=fname.endswith('.cif'),
                chain_in_filename=chain_in_filename)
