import os, shutil
import numpy as np
from scipy.interpolate import pchip
import json
import itertools
import copy
from bisect import bisect_left, insort
#######################

dft_TC_dir       = './DFT_TC_data'            # the folder to save DFT-json file
espei_dir        = './input_data_espei'       # the folder to save ESPEI-json file
input_model_name = 'INPUT+MODEL.json'         # model file in the working folder
comment1         = 'DFT done by shunli'       # any comment as character
reference1       = 'Unpublished shunli'       # any comment as character
weight1          = 1                          # weight for this kind of data
myT0             = np.arange(300, 910, 10)    # see below the note in print()

##---
print('Length of myT0 and the starting/ending T= ', len(myT0), myT0[0], myT0[-1])
##---
mix_phase_list = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3']       # used for later
###################################################################################################################
###################################################################################################################
###################################################################################################################
## define some funcitons below
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
def pickup_energy_from_json(data_dict1, myT0):              ## from the DFT-json file
    tt=[]; nhh=[]; nss=[]; ncp=[]; ncheck = 0
    ev2j = 1000 * 96.484542  # from eV to J
    unit1 = data_dict1.get("energy0_atom_unit")[-1][0].upper()      # use to check it is eV or not
    atom1 = data_dict1.get("energy0_atom_unit")[1]                  # per xx atom
    e0    = data_dict1.get("energy0_atom_unit")[0]
    if unit1 == 'E': # for case of eV
        e0 = e0 * ev2j / atom1;  # print('e0, type=', e0, type(e0)) # change to j/mol-atom
    else:
        e0 = e0 / atom1
    ##############
    if 'QHA_results' in data_dict1:  # pickup data for properties at finite T, when available
        ncheck = 1
        unit2=data_dict1.get("QHA_results").get("energy_units")[1].upper()
        atom2=data_dict1.get("QHA_results").get("energy_units")[0]
        hh   = np.asarray(data_dict1.get("QHA_results").get("H"))
        ss   = np.asarray(data_dict1.get("QHA_results").get("S"))
        tt   = np.asarray(data_dict1.get("QHA_results").get("T"))
        cp   = np.asarray(data_dict1.get("QHA_results").get("CP"))
        if unit2 == 'E': # for case of eV
            hh   = hh*ev2j / atom2 + e0;  ss = ss*ev2j / atom2;  cp = cp*ev2j / atom2
        else:
            #print('type hh', type(hh), 'type e0', type(e0), 'type atom2', type(atom2), 'type myT0', type(myT0))
            hh = hh / atom2 + e0;  ss = ss / atom2;  cp = cp / atom2
        nhh = pchip(tt,hh)(myT0)
        nss = pchip(tt,ss)(myT0)
        ncp = pchip(tt,cp)(myT0)

    return e0, tt, nhh, nss, ncp, ncheck

##------------
def To_write_json_file(in_dict, out_json_file):    # to write the ESPEI-json file
    with open(out_json_file, 'w') as f:
        asd2 = in_dict.get('solver').get('sublattice_configurations')
        asd3 = dedup(asd2)
        asd3.sort()
        f.write('{\n')
        f.write('  "comment": ')
        f.write(json.dumps(in_dict.get('comment')))
        f.write(',\n  "reference": ');               f.write(json.dumps(in_dict.get('reference')))
        f.write(',\n  "weight": ');                  f.write(json.dumps(in_dict.get('weight')))
        f.write(',\n  "phases":[');                  f.write(json.dumps(in_dict.get('phases')))
        f.write(']')
        f.write(',\n  "components": ');              f.write(json.dumps(asd3))
        f.write(',\n  "conditions": ');              f.write(json.dumps(in_dict.get('conditions')))
        f.write(',\n  "output": ');                  f.write(json.dumps(in_dict.get('output')))
        f.write(',\n  "solver": {')
        f.write( '\n      "mode": ', );                       f.write(json.dumps(in_dict.get('solver').get('mode')))
        f.write(',\n      "sublattice_site_ratios": ', );     f.write(json.dumps(in_dict.get('solver').get('sublattice_site_ratios')))
        if 'sublattice_occupancies' in  in_dict.get('solver'):
            f.write(',\n      "sublattice_occupancies": [', )
            f.write(json.dumps(in_dict.get('solver').get('sublattice_occupancies')))
            f.write(']')
        f.write(',\n      "sublattice_configurations":[ ', );  f.write(json.dumps(asd2))
        f.write('\n  ]}')
        f.write(',\n  "values": [')
        val1=in_dict.get('values')
        if type(val1) == float:
            #print('in: type1 float', type(val1))
            val2 = [[val1]]
        else:
            #print('in: type2 list', type(val1))
            val2 = [[x] for x in val1]
        f.write(json.dumps(val2))
        f.write("]")
        f.write('\n}')
        f.write('\n')
        #print('dict=', in_dict)
    f.close()

## -----------
def To_find_ene_refElem(ref_elem, my_phase, DFTTC_json_ref, myT0):
    e01=[]; tt1=[]; hh1=[]; ss1=[]; cp1=[]
    e02=[]; tt2=[]; hh2=[]; ss2=[]; cp2=[]
    nn1 = 0; nn2 =0; ncheck1=0; ncheck2 =0
    aaa = 'DFT_TC_data/reference_data/'
    for fname in DFTTC_json_ref:
        if fname[0:4] == 'DFT_' and fname[-4:] == 'json':
            with open(aaa + fname) as f:
                data_dict1 = json.load(f)
                serref = data_dict1.get("SER_refererce")[0].upper();  # print('s1=', serref)
                elem1  = data_dict1.get("elements_ratio_poscar");  # print('s2=', elem1)
                phase1 = data_dict1.get("phase_name")[0].upper()
                if len(elem1) == 1 and serref == 'Y' and elem1[0][0].upper() == ref_elem:  # Case 1 for FORM
                #if len(elem1) == 1 and elem1[0][0].upper() == ref_elem:  # Case 1 for FORM
                    nn1 = nn1 + 1
                    e01, tt1, hh1, ss1, cp1, ncheck1 = pickup_energy_from_json(data_dict1, myT0)  # prop of my_phase
                #if len(elem1) == 1 and phase1 == my_phase and elem1[0][0].upper() == ref_elem:  # Case 2 for MIX
                #    e02, tt2, hh2, ss2, cp2, ncheck2 = pickup_energy_from_json(data_dict1, myT0)  # prop of my_phase
                #    nn2 = nn2 + 1
    return e01, tt1, hh1, ss1, cp1, e02, tt2, hh2, ss2, cp2, ncheck1, ncheck2
    ## if nn1 > 1 or nn2 > 2, then, you give too many files for reference elements, report error, later do it.

###-------------------------
def find_total_elements_in_phase(elem3):
    # elem3 from DFT-json file:  elem3  = data_dict3.get("elements_ratio_poscar")
    nn = 0
    for i in range(len(elem3)):
        nn = nn + elem3[i][1]
    return nn

### ---------------
def find_solver_2_cases(sub_ratio, data_dict3):
    #sub_ratio  = model_dict.get('phases').get(my_phase).get('sublattice_site_ratios')
    #sub_modela = model_dict.get('phases').get(my_phase).get('sublattice_model')  # in model file
    #print('check: sub_ratio  =', sub_ratio)
    #print('check: sub_modela =', sub_modela)
    # data_dict3 is DFT-json file
    #total_atom_in_model = sub_ratio[0]
    asd =    data_dict3.get('sublattice').get('sublattice_occupancies')
    ccoonn = data_dict3.get('sublattice').get('sublattice_configurations')
    aa=0
    for i in range(len(asd)):
        if type(asd[i]) == list: aa = 1

    if aa == 0:  # solution case such as SQS, dilute, with sublattice_model inside
        solver = dict(mode='manual', sublattice_site_ratios = sub_ratio, \
                       sublattice_configurations = ccoonn)
    else:
        solver = dict(mode='manual', sublattice_site_ratios = sub_ratio, \
                       sublattice_configurations = ccoonn, sublattice_occupancies = asd)

    return solver

### END of functions
##########################################################################################
##########################################################################################
##########################################################################################
##
#dft_TC_dir       = './DFT_TC_data'            # the folder to save DFT-json file
#espei_dir        = './input_data_espei'       # the folder to save ESPEI-json file
#input_model_name = 'INPUT+MODEL.json'         # model file in the working folder
#comment1         = 'DFT done by shunli'       # any comment as character
#reference1       = 'Unpublished shunli'       # any comment as character
#weight1          = 1                          # weight for this kind of data
#myT0             = np.arange(300, 910, 10)    # see below the note in print()

print()
### some data/files to use
ev2j = 1000*96.484542   # from eV to J
conds1 = dict(P=101325, T=myT0.tolist())
conds298 = dict(P=101325, T=298.15)

if not os.path.exists(espei_dir):   #espei_dir        = './input_data_espei'       # the folder to save ESPEI-json file
    os.makedirs(espei_dir)
    print('Generate a folder to store ESPEI json file=', espei_dir)

##-------------
print('Search all DFT+*.json files in folder: ./DFT_TC_data: NOTE json in lower case')
DFTTC_json_files = os.listdir(dft_TC_dir)    #dft_TC_dir       = './DFT_TC_data'
DFTTC_json_ref = os.listdir(dft_TC_dir+'/reference_data')
#print('DFT_json_ref=',DFT_json_ref)
#print()
with open('./'+input_model_name) as f_file:    # open model-json file #input_model_name = 'INPUT+MODEL.json'
    model_dict = json.load(f_file)
    all_elem_model  = model_dict.get("components")
    all_phases      = model_dict.get("phases")
    for key in all_phases.keys():
        print('Phase in model json file =', key)
    print ('Elements in model json file = ', all_elem_model, len(all_elem_model))

##################################
print()
print('##### Do some calculations and then write ESPEI-json file(s) for each phase in model-json file')
for my_phase in all_phases.keys():     # all_phases in model-json file
    print()
    print()
    elem_all = all_phases.get(my_phase).get('sublattice_model')   # all_phases in model-json file
    elem_indepent =dedup([val for sublist in elem_all for val in sublist])
    no_elem_model = len(elem_indepent)
    sub_ratio  = model_dict.get('phases').get(my_phase).get('sublattice_site_ratios')
    sub_modela = model_dict.get('phases').get(my_phase).get('sublattice_model')  # in model file
    # print('sub_model a=', sub_modela, type(sub_modela), sub_modela[1][0])
    ## ---
    print('** Work for phase has elems =', my_phase, elem_all, ' ##Independ-elems & no.=', elem_indepent, no_elem_model)
    for fname in DFTTC_json_files:    #dft_TC_dir  = './DFT_TC_data'
        if fname[0:4] == 'DFT_' and fname[-4:] == 'json':
            name_end =fname[6:]; print('   -- check DFT-json file =', fname)
            with open(dft_TC_dir + '/' + fname) as f:
                data_dict3 = json.load(f)
                serref = data_dict3.get("SER_refererce")[0].upper();  # print('s1=', serref)
                elem3  = data_dict3.get("elements_ratio_poscar");  # print('xxxx====element3=', elem3)
                phase3 = data_dict3.get("phase_name")[0].upper()
                no_elem_phase = len(elem3)
                tot_elem_in_phase = find_total_elements_in_phase(elem3)

                #if phase3 == my_phase and no_elem_phase == no_elem_model:
                if phase3 == my_phase and serref != 'Y':   ## find the phase based on phase name
                    print('Find phase in DFT-json files, tot elems, file name = ', my_phase, tot_elem_in_phase, fname)
                    tot_atoms = 0; iwrite_json = -1
                    myT1 = [];    myS1 = [];   myH1 = [];    myCP1 = [];    myH01 = []
                    myT2 = [];    myS2 = [];   myH2 = [];    myCP2 = [];    myH02 = []
                    E03, TT3, HH3, SS3, CP3, ncheck0 = pickup_energy_from_json(data_dict3, myT0) # prop of my_phase
                    ## E03 is J/mol-atom of E0

                    # next for two cases with and without QHA_results for my_phase
                    if 'QHA_results' in data_dict3:
                        print('Case 1: for DFT results (not for elements) at finite temperatures')
                        num_elem = 0
                        for ref_elem in elem_indepent:  # to search all reference elements in model-json file
                            for k in range(len(elem3)):   # search elems in current DFT-json file
                                if elem3[k][0].upper() == ref_elem: num_elem = elem3[k][1]
                            if num_elem > 0:
                                print('Found a reference element with num_elem = ', ref_elem, num_elem)
                                tot_atoms = tot_atoms + num_elem
                                e01, tt1, hh1, ss1, cp1, e02, tt2, hh2, ss2, cp2, ncheck1, ncheck2 = \
                                    To_find_ene_refElem(ref_elem, my_phase, DFTTC_json_ref, myT0)
                                print('For ref_elem: e01, e02, & ncheck1, ncheck2 =', e01, e02, ncheck1, ncheck2)
                                #aaa=0.1; bbb=[]; print('aaaaxxx', type(aaa), type(bbb))
                                #print('======test type hh1 =', type(hh1))
                                if type(e01) != list and type(hh1) != list:  # for FORM, e01 and hh1 have values
                                    if type(myH01) == list:
                                        myH1  = num_elem * (HH3 - hh1);       myH01 = num_elem * (E03 - e01)
                                    else:
                                        myH1 = num_elem * (HH3 - hh1) + myH1; myH01 = num_elem * (E03 - e01) + myH01
                                    if type(myS1) == list:  myS1  = num_elem * (SS3 - ss1)
                                    else:                   myS1  = num_elem * (SS3 - ss1) + myS1
                                    if type(myCP1) == list: myCP1 = num_elem * (CP3 - cp1)
                                    else:                   myCP1 = num_elem * (CP3 - cp1) + myCP1

                                if type(e02) != list and type(hh2) != list: # for MIX, e02 ahd hh2 have values
                                    if type(myH02) == list:
                                        myH2 = num_elem * (HH3 - hh2);          myH02 = num_elem * (E03 - e02)
                                    else:
                                        myH2 = num_elem * (HH3 - hh2) + myH2;   myH02 = num_elem * (E03 - e02) + myH02
                                    if type(myS2) == list:   myS2  = num_elem * (SS3 - ss2)
                                    else:                    myS2  = num_elem * (SS3 - ss2) + myS2
                                    if type(myCP2) == list:  myCP2 = num_elem * (CP3 - cp2)
                                    else:                    myCP2 = num_elem * (CP3 - cp2) + myCP2
                    else:  # for the case without 'QHA_results'
                        print('Case 2: for DFT results (not for elements) at 0 K')
                        num_elem = 0
                        for ref_elem in elem_indepent:  # to search all reference elements in model-json file
                            for k in range(len(elem3)): # to search elems in current DFT-json file
                                if elem3[k][0].upper() == ref_elem:
                                    num_elem = elem3[k][1]
                                    if num_elem > 0:
                                        print('Found a reference element and num_elem = ', ref_elem, num_elem)
                                        tot_atoms = tot_atoms + num_elem
                                        e01, tt1, hh1, ss1, cp1, e02, tt2, hh2, ss2, cp2, ncheck1, ncheck2 = \
                                            To_find_ene_refElem(ref_elem, my_phase, DFTTC_json_ref, myT0)
                                        print('For ref_elem: e01, e02, & ncheck1, ncheck2=', e01, e02, ncheck1, ncheck2)
                                        if type(e01) != list:  # for FORM, e01 has a value
                                            if type(myH01) == list:    myH01 = num_elem * (E03 - e01)
                                            else:                      myH01 = num_elem * (E03 - e01) + myH01
                                        if type(e02) != list:  # for MIX, e01 has a value
                                            if type(myH02) == list:    myH02 = num_elem * (E03 - e02)
                                            else:                      myH02 = num_elem * (E03 - e02) + myH02

                    print ('For phase =', my_phase, 'e01_form =', e01, 'e02_mix =', e02, 'Total_elements =', tot_atoms)
                    if tot_atoms == tot_elem_in_phase:
                        print('Phase=', my_phase, 'has all reference elements, will write ESPEI-json file')

                    #print ('For phase =', my_phase, 'myH01_form =', myH01,  'myH02_form =', myH02)
                    print('-----')

                    ### =========================++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    print('Next to write ESPEI-json file for phase =', my_phase)
                    #print('The end part of this file =', name_end)

                    if tot_elem_in_phase == tot_atoms and 'QHA_results' in data_dict3:   # json for QHA case
                        # write ESPEI-json case 1: for both at finte T and at 0 K
                        solver1 = find_solver_2_cases(sub_ratio, data_dict3)
                        print('write ESPEI-json file for T at finite temperatures')
                        if type(myH1) != list:
                            prop_type = 'HM_FORM'
                            dictHH = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds1, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=(myH1 / tot_atoms).tolist())
                            outname = 'TT_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictHH, outname)
                        if type(myH01) != list:
                            prop_type = 'HM_FORM'
                            dictH0 = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds298, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=myH01/tot_atoms)
                            outname = 'T0_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictH0, outname)
                        if type(myH2) != list:
                            prop_type = 'HM_MIX'
                            dictHH = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds1, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=(myH2 / tot_atoms).tolist())
                            outname = 'TT_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictHH, outname)
                        if type(myH02) != list:
                            prop_type = 'HM_MIX'
                            dictH0 = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds298, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=myH02/tot_atoms)
                            outname = 'T0_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictH0, outname)

                        if type(myS1) != list:
                            prop_type = 'SM_FORM'
                            dictS  = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds1, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=(myS1 / tot_atoms).tolist())
                            outname = 'TT_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictS, outname)
                        if type(myS2) != list:
                            prop_type = 'SM_MIX'
                            dictS  = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds1, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=(myS2 / tot_atoms).tolist())
                            outname = 'TT_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictS, outname)

                        if type(myCP1) != list:
                            prop_type = 'CPM_FORM'
                            dictCP  = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds1, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=(myCP1 / tot_atoms).tolist())
                            outname = 'TT_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictCP, outname)
                        if type(myCP2) != list:
                            prop_type = 'CPM_MIX'
                            dictCP  = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds1, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=(myCP2 / tot_atoms).tolist())
                            outname = 'TT_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictCP, outname)

                    if tot_elem_in_phase == tot_atoms and 'QHA_results' not in data_dict3:  # json for case at 0 K
                        solver1 = find_solver_2_cases(sub_ratio, data_dict3)
                        print('write ESPEI-json file for T at 0 K')
                        if type(myH01) != list:
                            prop_type = 'HM_FORM'
                            dictH0 = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds298, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=myH01/tot_atoms)
                            outname = 'T0_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictH0, outname)
                        if type(myH02) != list:
                            prop_type = 'HM_MIX'
                            dictH0 = dict(components=elem_indepent, phases=my_phase, solver=solver1, \
                                          conditions=conds298, output=prop_type, reference=reference1, \
                                          comment=comment1, weight=weight1, values=myH02/tot_atoms)
                            #print('xxxxxxxxxxmyH02, tot_atoms=', myH02, tot_atoms, myH02/tot_atoms)
                            outname = 'T0_ESPEI+' + prop_type + '+' + name_end
                            To_write_json_file(dictH0, outname)
files = os.listdir()
print()
for f in files:
    if (f.startswith("T0_ESPEI+") or f.startswith("TT_ESPEI+")): #shutil.move(f, espei_dir)
        if os.path.isfile(espei_dir + '/' + f):
            print('  XXX File exists in folder =', espei_dir, f)
        else:
            shutil.move(f, espei_dir)
print()
print('#####    THE END    #####')
