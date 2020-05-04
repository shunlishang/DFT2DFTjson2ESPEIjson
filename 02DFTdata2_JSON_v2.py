import os, re, shutil, yaml
from subprocess import call
from bisect import bisect_left, insort
from numpy import *
import numpy as np
import json
import glob
import copy

###################
dft_TC_dir     = './DFT_TC_data'            # the folder to save DFT-json file
file_vasp_info = './vasp_info'              # the file contains DFT information
list_dir_name  = 'list.dir'                 # the dir names for each phase or endmember
run_my_script  = '../01short_get_info_v1'   # the linux script to get metadata from VASP
extra_name     = '+test'                # extra_info starting from "+" to be added in DFT-json file name

################################################################################################
## some functions below
##-----------
##
def dedup(seq):
    # Remove duplicates. Preserve order first seen. Assume orderable, but not hashable elements'
    result = []
    seen = []
    for x in seq:
        i = bisect_left(seen, x)
        if i == len(seen) or seen[i] != x:
            seen.insert(i, x)
            result.append(x)
    return result

##------------
def find_atom_num_poscar(atoms):
    aa0 =atoms[0].split(); #print(aa0)
    bb0 =atoms[1].split(); #print(bb0)
    bb1 = [int(x) for x in bb0]; #print('bb1 =', bb1)
    elem_unique = np.sort(np.unique(aa0));
    elem_list = elem_unique.tolist(); # print('atoms_unique_sort =', elem_list)
    atoms_sort = []
    for ii in range(len(elem_list)):
        nn = 0
        for jj in range(len(aa0)):
            if aa0[jj] == elem_list[ii]:
                nn = nn + bb1[jj]
        ccc =[elem_list[ii], nn]
        atoms_sort.append(ccc)
    #print('my_elem_number =', atoms_sort)
    return atoms_sort

## ---------------------------
def find_wyck_info(data):
    wyck_res = []
    asd = data.split('(WYCCAR)'); #print('wyck_data =', len(asd), asd[1])
    asd_lines = asd[1].split('\n'); # print('wyck_data =', len(asd_lines), asd_lines)
    for line in asd_lines:
        if line == '': continue
        filtered_line = [val for val in line.split(' ') if val != '']
        #asd2 = [float(filtered_line[0]), float(filtered_line[1]), float(filtered_line[2]), \
        #        filtered_line[3], int(filtered_line[4]), filtered_line[5]]
        asd2 = [filtered_line[3], int(filtered_line[4]), filtered_line[5]]
        wyck_res.append(asd2)
        #print('here =', filtered_line, asd2)
    #print('In Function wyck_res =', wyck_res)
    return wyck_res

##----------------------------
def extract_data_vasp_info(textvasp):
    #print('Function :', len(textvasp), type(textvasp))
    my_dict = {}; dict_vasp_set = {}; dict_QHA = {}

    ## ---- block 1
    block = re.split("phase_name_", textvasp)
    phases = block[1].split('\n')[1:-1];  #print(); print('block 1 for phases =', len(phases), phases)
    my_dict['phase_name'] = phases; #print(); print('block 1 for phases =', my_dict.get('phase_name'))

    ##---- block 2
    if 'wyccar_' in textvasp:
        block = re.split("wyccar_", textvasp)
        wyck = block[1].split('\n')[1:-1];  # print(); print('block 2 for Wyckoff =', len(wyck), wyck)
        aa = block[1].split('G:')[1];  # print();  print('block 2 for space group = ', aa)
        sginfo = aa.split(' ')[1:3];  # print(); print('block 2 for space group = ', sginfo)
        wyck_res = find_wyck_info(block[1]);  # print('block 2 for Wyckoff information =', wyck_res)
        my_dict['space_group'] = [sginfo[0], int(sginfo[1])]
        my_dict['Wyckoff_sites'] = wyck_res
        # print(); print('block 2 for SG and Wyckoff sites =', my_dict.get('space_group'), my_dict.get('Wyckoff_sites'))

    ##---- block 3
    if 'contcar_' in textvasp:
        block = re.split("contcar_", textvasp)
        atoms = block[1].split('\n')[1:-1];  # print(); print('block 3 for CONTCAR =', len(atoms), atoms)
        atoms_sort = find_atom_num_poscar(atoms);  # print('atoms_sort =', atoms_sort)
        my_dict['elements_ratio_poscar'] = atoms_sort
        # print();  print('block 3 from CONTCAR for atoms/number =', my_dict['elements_ratio_poscar'])

    ##---- block 4
    if 'eos_0k_' in textvasp:
        block = re.split("eos_0k_", textvasp)
        eos = block[1].split('\n')[1:-1];  # print(); print('block 4 for EOS at 0 K =', len(eos), eos)
        eos_res = [float(x) for x in eos[0].split()]
        my_dict['eos_properties'] = eos_res
        my_dict['energy0_atom_unit'] = [eos_res[1], 1, 'eV'];  # my_dict['energy0_atom_unit'][1] = 110
        # print(); print('block 4 for EOS and energy0_atom_unit =', my_dict['eos_properties'], my_dict['energy0_atom_unit'])

    ##---- block 5
    if 'potcar_' in textvasp:
        block = re.split("potcar_", textvasp)
        pots = block[1].split('\n')[1:-1];  # print(); print('block 5 for POTCAR =', len(pots), pots)
        pots_list = []
        for i in range(len(pots)):
            aa = pots[i].split(' = ')[1];  # print('pot_here =', aa)
            pots_list.append(aa)
        # my_dict['POTCAR'] = pots_list
        # print(); print('block 5 for POTCAR =', pots_list)

    ##---- block 6
    if 'kpoints_' in textvasp:
        block = re.split("kpoints_", textvasp)
        kps = block[1].split('\n')[1:-1];  # print(); print('block 6 for KPOINTS =', len(kps), kps)
        if kps[0][0].upper() == 'G':
            aa0 = 'Gamma'
        if kps[0][0].upper() == 'M':
            aa0 = 'M-P'
        if kps[0][0].upper() == 'A':
            aa0 = 'Auto'
        aa1 = [int(x) for x in kps[1].split()]
        my_kps = [aa0, aa1]
        # print(); print('block 6 for KPOINTS =', my_kps)

    ##---- block 7
    if 'incar_' in textvasp:
        block = re.split("incar_", textvasp)
        vasp_set = block[1].split('\n')[1:-1];  # print(); print('block 7 for VASP Settings =', len(vasp_set), vasp_set)
        my_vasp = [];
        for i in range(len(vasp_set)):
            asd0 = vasp_set[i].split();  # print(asd0)
            my_vasp.append([asd0[0], int(asd0[2])])
        # print('my_vasp_setting =', my_vasp)
        for i in range(len(my_vasp)):
            dict_vasp_set[my_vasp[i][0]] = my_vasp[i][1]
        dict_vasp_set['KPOINTS'] = my_kps
        dict_vasp_set['POTCAR'] = pots_list
        # print(); print('blocks 5, 6, 7 for VASP Settings =', dict_vasp_set)
        my_dict['VASP_settings'] = dict_vasp_set

    ##---- block 8
    block = re.split("path_atom_", textvasp)
    paths = block[1].split('\n')[1:-1];  # print(); print('block 8 for PATH =', len(paths), paths)
    my_path = paths[0];
    my_dict['folder_name'] = my_path;  #print(); print('block 8a for PATH =', my_dict['folder_name'])
    if os.path.exists('./TC'):
        aaa = paths[1].split()
        atoms_in_QHA = re.sub("\D", "", aaa[2])
        #print('atoms used in QHA =', atoms_in_QHA)

    ##----- block 9/10
    block = re.split("note_", textvasp)
    note0 = block[1].split('\n')[1:-1]; #print(); print('block 9/10 for NOTE =', len(note0), note0)
    my_dict['notes'] = note0; #print(); print('block 9/10 for NOTE =', my_dict['notes'])
    if os.path.exists('temp.h') and os.path.exists('temp.t'):
        print('for QHA case')
        dict_QHA['methods'] = ['phonon', 'thermal_ele']
        dict_QHA['energy_units'] = [int(atoms_in_QHA), 'J', 'mole-atom', 'K']
        with open("temp.t") as f:
            my_data = f.read().split(', ')
            my_tt   = [float(x) for x in my_data];  #print('my_tt =', my_tt)
            dict_QHA['T'] = my_tt; #print('my_tt =', dict_QHA['T'])
        with open("temp.h") as f:
            my_data = f.read().split(', ')
            my_hh   = [float(x) for x in my_data]
            dict_QHA['H'] = my_hh
        with open("temp.s") as f:
            my_data = f.read().split(', ')
            my_ss   = [float(x) for x in my_data]
            dict_QHA['S'] = my_ss
        with open("temp.cp") as f:
            my_data = f.read().split(', ')
            my_cp   = [float(x) for x in my_data]
            dict_QHA['CP'] = my_cp
        my_dict['QHA_results'] = dict_QHA
        #print('QHA data len tt, hh, ss, cp =', len(my_tt), len(my_hh), len(my_ss), len(my_cp))

    ##---- block 11
    block = re.split("SER_ref_", textvasp)
    ser0 = block[1].split('\n')[1:-1];
    my_dict['SER_refererce'] = ser0[0]; # print(); print('block 11 for SER_ref =', my_dict['SER_refererce'])
    #print('*** end of function to create my_dict ***');  print()

    return my_dict

## ---------------
def To_write_DFT_json_file(in_dict, out_json_file):
    with open(out_json_file, 'w') as f:
        f.write('{\n')
        f.write('  "phase_name": ');                 f.write(json.dumps(in_dict.get('phase_name')))
        f.write(',\n  "elements_ratio_poscar": ');   f.write(json.dumps(in_dict.get('elements_ratio_poscar')))
        f.write(',\n  "SER_refererce": ');           f.write(json.dumps(in_dict.get('SER_refererce')))
        f.write(',\n  "space_group": ');             f.write(json.dumps(in_dict.get('space_group')))
        f.write(',\n  "Wyckoff_sites": ');           f.write(json.dumps(in_dict.get('Wyckoff_sites')))
        f.write(',\n  "energy0_atom_unit": ');       f.write(json.dumps(in_dict.get('energy0_atom_unit')))
        f.write(',\n  "VASP_settings": ');           f.write(json.dumps(in_dict.get('VASP_settings'), indent=2))
        f.write(',\n  "eos_properties": ');          f.write(json.dumps(in_dict.get('eos_properties')))
        f.write(',\n  "folder_name": ');             f.write(json.dumps(in_dict.get('folder_name')))
        f.write(',\n  "notes": ');                   f.write(json.dumps(in_dict.get('notes')))
        f.write(',\n  "sublattice": ');              f.write(json.dumps(in_dict.get('sublattice')))
        if 'QHA_results' in in_dict:
            f.write(",\n")
            f.write('\n  "QHA_results": {')
            f.write('\n    "methods": ');       f.write(json.dumps(in_dict.get('QHA_results').get('methods')))
            f.write(',\n    "energy_units": '); f.write(json.dumps(in_dict.get('QHA_results').get('energy_units')))
            f.write(',\n    "T":  ');           f.write(json.dumps(in_dict.get('QHA_results').get('T')))
            f.write(',\n    "S":  ');           f.write(json.dumps(in_dict.get('QHA_results').get('S')))
            f.write(',\n    "H":  ');           f.write(json.dumps(in_dict.get('QHA_results').get('H')))
            f.write(',\n    "CP": ');           f.write(json.dumps(in_dict.get('QHA_results').get('CP')))
            f.write('\n  }')
        f.write('\n}')
        f.write('\n')
    f.close()

## end of functions
###############################################################################################
###############################################################################################
####

#dft_TC_dir     = './DFT_TC_data'            # the folder to save DFT-json file
#file_vasp_info = './vasp_info'              # the file contains DFT information
#list_dir_name  = 'list.dir'                 # the dir names for each phase or endmember
#run_my_script  = '../01short_get_info_v1'   # the linux script to get metadata from VASP
#extra_name     = '+CrHftest'                # extra_info starting from "+" to be added in DFT-json file name

if not os.path.exists(dft_TC_dir):    # usually './DFT_TC_data' 
    os.makedirs(dft_TC_dir)
    print('Generate a folder =', dft_TC_dir)
if os.path.exists(list_dir_name):               # usually './list.dir'
    my_dir0  = open(list_dir_name, 'r')         # open file to read: dir_folder names
    textdir0 = my_dir0.read().split('\n')       # read dir_folder names
    print('dir name =', textdir0, len(textdir0))
    if textdir0[-1] == '':
        #print('Yes, the last folder name is empty')
        mylen =len(textdir0) - 1
    else:
        mylen = len(textdir0)
    for idir in range(mylen):
        os.chdir(textdir0[idir])
        print()
        print('Get data in folder =', textdir0[idir])
        fileyaml='sub_conf_occ.yaml'
        if os.path.exists(fileyaml):
            dict_sublattice={}
            config=[];
            occupy=[];
            asd = 0
            with open(fileyaml) as f:
                data1 = yaml.load(f, Loader=yaml.FullLoader)
                subconf=data1.get('sub_configurations')
                print('subconf====',subconf, len(subconf))
                asdx=[];
                for i in range(len(subconf)):
                    if type(subconf[i]) == list:
                        yy   = [x.upper() for x in subconf[i]]
                        asdx.append(yy)
                    else:
                        asdx.append(subconf[i].upper())
                #subconf = [x.upper() for x in subconf]
                subconf = copy.deepcopy(asdx)

                suboccu=data1.get('sub_occupancies')
                numsub = len(subconf)
                for ii in range(numsub):
                    config.append(subconf[ii])
                    if type(subconf[ii]) != list:
                        occupy.append(1)
                    else:
                        asd = 1
                        occupy.append(suboccu[ii])
                dict_sublattice['sublattice_occupancies']    = occupy
                dict_sublattice['sublattice_configurations'] = config
        ##############################################################
        if os.path.exists(file_vasp_info):
            my_file = open(file_vasp_info, 'r')
        else:
            os.system('sh ' + run_my_script)
            my_file  = open(file_vasp_info,'r')
        textvasp = my_file.read() #.split('_start')
        my_dict = extract_data_vasp_info(textvasp);  #print(); print('my_dict =', my_dict)
        my_dict['sublattice'] = dict_sublattice
        name1 = my_dict.get('folder_name').split('/')[-1]; #print('name1 = ', name1)
        if 'QHA_results' in my_dict:
            out_json_name = 'DFT_T+' + my_dict['phase_name'][0] + '+' + name1 + extra_name + '.json'
        else:
            out_json_name = 'DFT_0+' + my_dict['phase_name'][0] + '+' + name1 + extra_name + '.json'
        To_write_DFT_json_file(my_dict, out_json_name)
        shutil.move(out_json_name, "." + dft_TC_dir  + "/" + out_json_name)

        os.chdir('..')
print()
print('#### THE END OF THE MAIN CODE ####')
print()



