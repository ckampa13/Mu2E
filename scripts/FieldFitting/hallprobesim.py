#! /usr/bin/env python

from __future__ import absolute_import
from mu2e.hallprober import field_map_analysis
from collections import namedtuple
import numpy as np
from mu2e import mu2e_ext_path
from mu2e.src.make_csv import make_csv
from six.moves import range

############################
# defining the cfg structs #
############################

cfg_data   = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom   = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps x_steps y_steps '
                        'bad_calibration interpolate do2pi')
cfg_params = namedtuple('cfg_params', 'ns ms cns cms nms Reff n_scale m_scale func_version')
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')
cfg_plot   = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir')

syst_set = 20

#################
# the data cfgs #
#################
path_PS_Mau10       = mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_PSMap'
path_DS_Mau10       = mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap'
path_DS_Mau13       = mu2e_ext_path+'datafiles/Mau13/Mu2e_DSMap_V13'
path_DSoffset_Mau10 = mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap_8mmOffset'
path_DS_GA05        = mu2e_ext_path+'datafiles/FieldMapsGA05/DSMap'
path_DSoffset_GA05  = mu2e_ext_path+'datafiles/FieldMapsGA05/DSMap_offset8mm'
path_DSnoext_GA05   = mu2e_ext_path+'datafiles/FieldMapsGA_Special/Mu2e_DS_noPSTS_GA0'
path_DSnoDS_GA05    = mu2e_ext_path+'datafiles/FieldMapsGA_Special/Mu2e_DS_noDS_GA0'
path_DS_GA02        = mu2e_ext_path+'datafiles/FieldMapsGA02/Mu2e_DS_GA0'
path_DS_Cyl_Only_Glass    = mu2e_ext_path+'datafiles/FieldMapsPure/DS_TS5_ideal_fullmap'
path_DS_Bus_Only_Glass    = mu2e_ext_path+'datafiles/FieldMapsPure/DS_longbus_nocoils'
path_DS_Bus_Rot_Glass    = mu2e_ext_path+'datafiles/FieldMapsPure/DS_buswork_only_rot'
path_DS_Combo_Glass    = mu2e_ext_path+'datafiles/FieldMapsPure/TS5_DS_longbus'
path_DS_Hel_Only_Glass    = mu2e_ext_path+'datafiles/FieldMapsPure/DS-8_with_leads'
path_DS_Hel_Only_Glass_2    = mu2e_ext_path+'datafiles/FieldMapsPure/DS-8_helix_no_leads'
path_DS_Hel_Only_Glass_3    = mu2e_ext_path+'datafiles/FieldMapsPure/test_helix_5L'
path_DS_Hel_Only_Glass_4    = mu2e_ext_path+'datafiles/FieldMapsPure/test_helix_10cm_pitch_tol6'
path_DS_Hel_Only_Glass_3_mod = mu2e_ext_path+'datafiles/FieldMapsPure/test_helix_5L_mod'
path_synth          = mu2e_ext_path+'datafiles/synth/synth3'
path_DS_Hel_Only_Cole = mu2e_ext_path+'datafiles/FieldMapsCole/bfield_map_3453103pts_06-29_231454'
path_DS_Hel_True_Cole = mu2e_ext_path +\
    'datafiles/FieldMapsCole/bfield_map_cylin_845568pts_07-03_145644'
path_DS_Hel_True_Cole2 = mu2e_ext_path +\
    'datafiles/FieldMapsCole/bfield_map_cylin_845568pts_07-06_160144'
path_DS_Hel_Only_Cole2 = mu2e_ext_path +\
    'datafiles/FieldMapsCole/bfield_map_1232173pts_08-10_094943'
path_DS_Hel_Only_Cole3 = mu2e_ext_path +\
    'datafiles/FieldMapsCole/bfield_map_r250mm_p10cm_lengthx10_1232173pts_09-07_160736'
path_DS_Hel_Only_Cole4 = mu2e_ext_path +\
    'datafiles/FieldMapsCole/10x_bfield_map_cylin_985152pts_09-20_162454'
path_DS_Hel_Nubs = mu2e_ext_path +\
    'datafiles/FieldMapsCole/endsonly_both_3453103pts_08-28_184114'

# PS
cfg_data_PS_Mau10        = cfg_data('Mau10', 'PS', path_PS_Mau10,
                                    ('Z>-7900', 'Z<-4000', 'R!=0'))

# Mau10
cfg_data_DS_Mau10_offset = cfg_data('Mau10', 'DS', path_DSoffset_Mau10,
                                    ('Z>5000', 'Z<13000', 'R!=0'))
cfg_data_DS_Mau10        = cfg_data('Mau10', 'DS', path_DS_Mau10,
                                    ('Z>5000', 'Z<13000', 'R!=0'))
cfg_data_DS_Mau10_long   = cfg_data('Mau10', 'DS', path_DS_Mau10,
                                    ('Z>4200', 'Z<13900', 'R!=0'))
cfg_data_DS2_Mau10       = cfg_data('Mau10', 'DS', path_DS_Mau10,
                                    ('Z>4000', 'Z<14000', 'R!=0'))
# Mau13
cfg_data_DS_Mau13        = cfg_data('Mau13', 'DS', path_DS_Mau13,
                                    ('Z>4200', 'Z<13900', 'R!=0'))

# GA05
cfg_data_DS_GA05_no_ext  = cfg_data('GA05', 'DS', path_DSnoext_GA05,
                                    ('Z>5000', 'Z<13000', 'R!=0'))
cfg_data_DS_GA05_no_DS   = cfg_data('GA05', 'DS', path_DSnoDS_GA05,
                                    ('Z>5000', 'Z<13000', 'R!=0'))
cfg_data_DS_GA05_offset  = cfg_data('GA05', 'DS', path_DSoffset_GA05,
                                    ('Z>5000', 'Z<13000', 'R!=0'))
cfg_data_DS_GA05         = cfg_data('GA05', 'DS', path_DS_GA05,
                                    ('Z>4200', 'Z<13900', 'R!=0'))
cfg_data_DS_GA05_grad    = cfg_data('GA05', 'DS', path_DS_GA05,
                                    ('Z>4200', 'Z<7000', 'R!=0'))
cfg_data_DS_GA02         = cfg_data('GA02', 'DS', path_DS_GA02,
                                    ('Z>4000', 'Z<14000', 'R!=0'))
cfg_data_DS_GA05_seg_trk = cfg_data('GA05', 'DS', path_DS_GA05,
                                    ('Z>8300', 'Z<12700', 'R!=0'))
cfg_data_DS_GA05_seg_trk2 = cfg_data('GA05', 'DS', path_DS_GA05,
                                     ('Z>9900', 'Z<11400', 'R!=0'))
# Glass
cfg_data_DS_Glass_Cyl     = cfg_data('Glass', 'DS', path_DS_Cyl_Only_Glass,
                                     ('Z>4500', 'Z<13500', 'R!=0'))
cfg_data_DS_Glass_Bus     = cfg_data('Glass', 'DS', path_DS_Bus_Only_Glass,
                                     ('Z>4000', 'Z<13500'))
cfg_data_DS_Glass_Combo   = cfg_data('Glass', 'DS', path_DS_Combo_Glass,
                                     ('Z>4500', 'Z<13500', 'R!=0'))
cfg_data_DS_Glass_Bus_Rot = cfg_data('Glass', 'DS', path_DS_Bus_Rot_Glass,
                                     ('Z>4000', 'Z<13500'))
cfg_data_DS_Glass_Hel     = cfg_data('Glass', 'DS', path_DS_Hel_Only_Glass,
                                     ('Z>8100', 'Z<9500', 'R!=0'))
cfg_data_DS_Glass_Hel_2   = cfg_data('Glass', 'DS', path_DS_Hel_Only_Glass_2,
                                     ('Z>8100', 'Z<9500', 'R!=0'))
cfg_data_DS_Glass_Hel_3   = cfg_data('Glass', 'DS', path_DS_Hel_Only_Glass_3,
                                     ('Z>-3500', 'Z<3500', 'R!=0'))
cfg_data_DS_Glass_Hel_3_mod = cfg_data('Glass', 'DS', path_DS_Hel_Only_Glass_3_mod,
                                       ('Z>-3500', 'Z<3500', 'R!=0'))
cfg_data_DS_Glass_Hel_3L  = cfg_data('Glass', 'DS', path_DS_Hel_Only_Glass_3,
                                     ('Z>-4500', 'Z<4500', 'R!=0'))
cfg_data_DS_Glass_Hel_3S  = cfg_data('Glass', 'DS', path_DS_Hel_Only_Glass_3,
                                     ('Z>500', 'Z<3500', 'R!=0'))
cfg_data_DS_Glass_Hel_4   = cfg_data('Glass', 'DS', path_DS_Hel_Only_Glass_4,
                                     ('Z>-3500', 'Z<3500', 'R!=0'))
cfg_data_synth  = cfg_data('synth', 'DS', path_synth,
                           ('Z>-3500', 'Z<4500', 'R!=0'))

# Cole
cfg_data_DS_Cole_Hel   = cfg_data('Cole', 'DS', path_DS_Hel_Only_Cole,
                                  ('Z>-3500', 'Z<3500', 'R!=0'))
cfg_data_DS_Cole_Hel_10x  = cfg_data('Cole', 'DS', path_DS_Hel_Only_Cole3,
                                     ('Z>-3.5', 'Z<3.5', 'R!=0'))
cfg_data_DS_Cole_Hel_10x2  = cfg_data('Cole', 'DS', path_DS_Hel_Only_Cole4,
                                      ('Z>-3.5', 'Z<3.5', 'R!=0'))
cfg_data_DS_Cole_Hel_true   = cfg_data('Cole', 'DS', path_DS_Hel_True_Cole,
                                       ('Z>-3500', 'Z<3500', 'R!=0'))
cfg_data_DS_Cole_Hel_descale = cfg_data('Cole', 'DS', path_DS_Hel_True_Cole2,
                                        ('Z>-1.5', 'Z<1.5', 'R!=0'))
cfg_data_DS_Cole_Nubs_descale = cfg_data('Cole', 'DS', path_DS_Hel_Nubs,
                                         ('Z>-1.5', 'Z<1.5', 'R!=0'))

# ('Z>6400', 'Z<11200', 'R!=0'))

#################
# the geom cfgs #
#################
# DS
pi8r_600mm = [55.90169944, 167.70509831, 279.50849719, 447.2135955, 614.91869381]
pi4r_600mm = [35.35533906, 141.42135624, 318.19805153, 494.97474683, 601.04076401]
pi2r_600mm = [25, 150, 325, 475, 600]

pi8r_700mm = [55.90169944, 167.70509831, 335.41019663, 559.01699437, 726.72209269]
pi4r_700mm = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 707.10678119]
pi2r_700mm = [25, 175, 375, 525, 700]

pi8r_825mm_v1 = [55.90169944, 167.70509831, 335.41019663, 559.01699437, 838.52549156]
pi4r_825mm_v1 = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 813.17279836]
pi2r_825mm_v1 = [25, 175, 375, 525, 825]

pi8r_825mm_v2 = [55.90169944, 335.41019663, 503.11529494, 726.72209269, 838.52549156]
pi4r_825mm_v2 = [35.35533906, 318.19805153, 494.97474683, 707.10678119, 813.17279836]
pi2r_825mm_v2 = [25, 325, 500, 725, 825]


pi8r_800mm = [55.90169944, 167.70509831, 335.41019663, 559.01699437, 782.62379213]
pi4r_800mm = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 813.17279836]
pi2r_800mm = [25, 175, 375, 525, 800]

pi8r_800mm_sml = [0.05590169944, 0.16770509831, 0.33541019663, 0.55901699437, 0.78262379213]
pi4r_800mm_sml = [0.03535533906, 0.1767766953, 0.35355339059, 0.53033008589, 0.81317279836]
pi2r_800mm_sml = [0.025, 0.175, 0.375, 0.525, 0.800]

pi8r_250mm = [0.02795084972, 0.05590169944, 0.08385254916, 0.11180339887, 0.16770509831]
pi4r_250mm = [0.01767766953, 0.05303300859, 0.08838834765, 0.10606601718, 0.15909902577]
pi2r_250mm = [0.0125, 0.050, 0.0875, 0.125, 0.150]

piall_250mm = [0.0125, 0.0375, 0.0625, 0.0875, 0.1125, 0.150]

pi8r_fullsim = [55.90169944, 111.80339887, 167.70509831, 223.60679775, 279.50849719, 335.41019663,
                391.31189606, 447.2135955, 503.11529494, 559.01699437, 614.91869381, 670.82039325,
                726.72209269, 782.62379213, 838.52549156, 894.427191, 1006.23058987]
pi4r_fullsim = [35.35533906, 106.06601718, 176.7766953, 247.48737342, 282.84271247, 318.19805153,
                388.90872965, 424.26406871, 494.97474683, 565.68542495, 601.04076401, 671.75144213,
                742.46212025, 777.81745931, 813.17279836, 883.88347648, 989.94949366]
pi2r_fullsim = [25, 75, 125, 175, 225, 275, 325, 400, 475, 525, 575, 625, 675, 725, 800, 900, 1000]

pi8r_fullsim800 = [55.90169944, 111.80339887, 167.70509831, 223.60679775, 279.50849719,
                   335.41019663, 391.31189606, 447.2135955, 503.11529494, 559.01699437,
                   614.91869381, 670.82039325, 726.72209269, 782.62379213]
pi4r_fullsim800 = [35.35533906, 106.06601718, 176.7766953, 247.48737342, 282.84271247, 318.19805153,
                   388.90872965, 424.26406871, 494.97474683, 565.68542495, 601.04076401,
                   671.75144213, 742.46212025, 813.17279836]
pi2r_fullsim800 = [25, 75, 125, 175, 225, 275, 325, 400, 475, 525, 575, 625, 700, 800]

pi2r_offset = [33, 133, 333, 533, 783, 42, 142, 342, 542, 792]

piall_800mm = [25, 175, 375, 525, 800]

# PS
pi8r_150mm = [55.90169944, 111.80339887, 167.70509831]
pi4r_150mm = [35.35533906, 106.06601718, 141.42135624]
pi2r_150mm = [25, 100, 150]

pi8r_150mm = [55.90169944, 111.80339887, 167.70509831]
pi4r_150mm = [35.35533906, 106.06601718, 141.42135624]
pi2r_150mm = [25, 100, 150]

r_steps_150mm = (pi2r_150mm, pi8r_150mm, pi4r_150mm, pi8r_150mm,
                 pi2r_150mm, pi8r_150mm, pi4r_150mm, pi8r_150mm)
r_steps_250mm = (pi2r_250mm, pi8r_250mm, pi4r_250mm, pi8r_250mm,
                 pi2r_250mm, pi8r_250mm, pi4r_250mm, pi8r_250mm)
r_steps_600mm = (pi2r_600mm, pi8r_600mm, pi4r_600mm, pi8r_600mm,
                 pi2r_600mm, pi8r_600mm, pi4r_600mm, pi8r_600mm)
r_steps_700mm = (pi2r_700mm, pi8r_700mm, pi4r_700mm, pi8r_700mm,
                 pi2r_700mm, pi8r_700mm, pi4r_700mm, pi8r_700mm)

r_steps_825mm_v1 = (pi2r_825mm_v1, pi8r_825mm_v1, pi4r_825mm_v1, pi8r_825mm_v1,
                    pi2r_825mm_v1, pi8r_825mm_v1, pi4r_825mm_v1, pi8r_825mm_v1)
r_steps_825mm_v2 = (pi2r_825mm_v2, pi8r_825mm_v2, pi4r_825mm_v2, pi8r_825mm_v2,
                    pi2r_825mm_v2, pi8r_825mm_v2, pi4r_825mm_v2, pi8r_825mm_v2)

r_steps_800mm = (pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm,
                 pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm)

r_steps_800mm_sml = (pi2r_800mm_sml, pi8r_800mm_sml, pi4r_800mm_sml, pi8r_800mm_sml,
                     pi2r_800mm_sml, pi8r_800mm_sml, pi4r_800mm_sml, pi8r_800mm_sml)

r_steps_fullsim = (pi2r_fullsim, pi8r_fullsim, pi4r_fullsim, pi8r_fullsim,
                   pi2r_fullsim, pi8r_fullsim, pi4r_fullsim, pi8r_fullsim)
r_steps_fullsim_trunc = (pi2r_fullsim800, pi8r_fullsim800, pi4r_fullsim800, pi8r_fullsim800,
                         pi2r_fullsim800, pi8r_fullsim800, pi4r_fullsim800, pi8r_fullsim800)

r_steps_800mm_true = (piall_800mm,)*8
r_steps_250mm_true = (piall_250mm,)*8


phi_steps_8 = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444, 3*np.pi/4, 2.677945)
phi_steps_true = (0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8)

z_steps_DS = list(range(5021, 13021, 50))
z_steps_DS_long = list(range(4221, 13921, 50))
z_steps_DS_trk = list(range(9021, 11021, 50))
z_steps_DS_grad = list(range(4221, 6921, 100))
z_steps_DS_interp = list(range(4220, 13900, 100))
z_steps_DS_interp_fullsim = list(range(4200, 14000, 25))
z_steps_DS_interp_v2 = list(range(4200, 8400, 50))+list(range(8400, 14000, 100))
z_steps_DS_less = list(range(9021, 11021, 50))
z_steps_DS_20cm = list(range(5021, 13021, 200))
z_steps_DS_30cm = list(range(5021, 13021, 300))
z_steps_DS_40cm = list(range(5021, 13021, 400))
z_steps_DS_50cm = list(range(5021, 13021, 500))
z_steps_DS2 = list(range(4171, 13921, 50))
z_steps_DS_fullsim = list(range(5021, 13021, 25))
z_steps_DS_fullsim2 = list(range(4221, 13921, 25))
z_steps_PS = list(range(-7879, -4004, 50))
z_steps_DS_seg_trk = list(range(8371, 12621, 50))
z_steps_DS_seg_trk2 = list(range(9921, 11371, 50))
z_steps_DS_glass = list(range(4021, 13521, 100))
z_steps_DS_glass_hel = list(range(8121, 9521, 25))
z_steps_DS_glass_test = list(range(-3500, 3500, 25))
z_steps_DS_glass_small = [i*1e-3 for i in range(-1500, 1500, 25)]
# z_steps_DS_glass_hel = list(range(6421, 11221, 25))

x_steps_DS_glass = list(range(-800, 801, 200))
y_steps_DS_glass = [-300, -150, -50, 0, 50, 150, 300]

x_steps_DS_glass_rot = [-795.495129, -583.363094, -406.586399, -194.454365, -17.67767, 17.67767,
                        194.454365, 406.586399, 583.363094, 795.495129]
y_steps_DS_glass_rot = [-300.520382, -159.099026, -53.033009, 53.033009, 159.099026, 300.520382]

# for interp
phi_steps_interp = [(i/8.0)*np.pi for i in range(0, 8)]
r_steps_interp = [[25, 225, 425, 625, 800]]*8
# r_set_interp = [25, 225, 425, 625]

# DS
cfg_geom_cyl_600mm              = cfg_geom('cyl', z_steps_DS_less, r_steps_600mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_700mm              = cfg_geom('cyl', z_steps_DS, r_steps_700mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_825mm_v1           = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v1, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_825mm_v2           = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v2, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm              = cfg_geom('cyl', z_steps_DS, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_long         = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_trk          = cfg_geom('cyl', z_steps_DS_trk, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_grad         = cfg_geom('cyl', z_steps_DS_grad, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

# Interp!!
# Interp!!
# Interp!!
cfg_geom_cyl_800mm_interp       = cfg_geom('cyl', z_steps_DS_interp, r_steps_interp,
                                           phi_steps_interp, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False],
                                           interpolate='load1', do2pi=False)

cfg_geom_cyl_800mm_interp_v2    = cfg_geom('cyl', z_steps_DS_interp_v2, r_steps_interp,
                                           phi_steps_interp, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False],
                                           interpolate='load2', do2pi=False)

cfg_geom_cyl_800mm_interp_v3    = cfg_geom('cyl', z_steps_DS_interp, r_steps_interp,
                                           phi_steps_interp, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False],
                                           interpolate='load3', do2pi=False)

cfg_geom_cyl_800mm_slice        = cfg_geom('cyl', z_steps_DS, r_steps_800mm[0:1], phi_steps_8[0:1],
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_600mm_slice        = cfg_geom('cyl', z_steps_DS_less, r_steps_600mm[0:1],
                                           phi_steps_8[0:1], x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_20cm         = cfg_geom('cyl', z_steps_DS_20cm, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_30cm         = cfg_geom('cyl', z_steps_DS_30cm, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_40cm         = cfg_geom('cyl', z_steps_DS_40cm, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_50cm         = cfg_geom('cyl', z_steps_DS_50cm, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_fullsim            = cfg_geom('cyl', z_steps_DS_fullsim, r_steps_fullsim, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_fullsim_trunc      = cfg_geom('cyl', z_steps_DS_fullsim, r_steps_fullsim_trunc,
                                           phi_steps_8, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_fullsim_trunc_test = cfg_geom('cyl', z_steps_DS_fullsim, r_steps_fullsim_trunc,
                                           phi_steps_8[:1], x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_fullsim2           = cfg_geom('cyl', z_steps_DS_fullsim2, r_steps_fullsim, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_fullsim800         = cfg_geom('cyl', z_steps_DS_fullsim2, r_steps_fullsim_trunc,
                                           phi_steps_8, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_measure_v1     = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v1, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[True, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_position_v1    = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v1, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, True, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_rotation_v1    = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v1, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, True], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_rotation2_v1   = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v1, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, True], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_measure_v2     = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v2, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[True, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_position_v2    = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v2, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, True, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_rotation_v2    = cfg_geom('cyl', z_steps_DS, r_steps_825mm_v2, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, True], interpolate=False,
                                           do2pi=False)

# PS
cfg_geom_cyl_150mm              = cfg_geom('cyl', z_steps_PS, r_steps_150mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_measure_req    = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[True, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_position_req   = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, True, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_rotation_req   = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, True], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_bad_all_req        = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[True, True, True], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_800mm_interp_slice = cfg_geom('cyl', z_steps_DS, r_steps_interp[0:1],
                                           phi_steps_interp[0:1], x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_offset             = cfg_geom('cyl', z_steps_DS, [pi2r_offset], [0], x_steps=None,
                                           y_steps=None, bad_calibration=[False, False, False],
                                           interpolate=False, do2pi=False)

cfg_geom_cyl_seg_trk            = cfg_geom('cyl', z_steps_DS_seg_trk, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_seg_trk2           = cfg_geom('cyl', z_steps_DS_seg_trk2, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_cyl_glass              = cfg_geom('cyl', z_steps_DS_glass, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_hel_glass              = cfg_geom('cyl', z_steps_DS_glass_hel, r_steps_fullsim_trunc,
                                           phi_steps_8, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_hel_glass_test         = cfg_geom('cyl', z_steps_DS_glass_test, r_steps_800mm[0:],
                                           phi_steps_8[0:], x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_hel_glass_small        = cfg_geom('cyl', z_steps_DS_glass_small, r_steps_800mm[0:],
                                           phi_steps_8[0:], x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_bus_glass              = cfg_geom('cart', z_steps_DS_glass, r_steps=None, phi_steps=None,
                                           x_steps=x_steps_DS_glass, y_steps=y_steps_DS_glass,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_bus_rot_glass          = cfg_geom('cart', z_steps_DS_glass, r_steps=None, phi_steps=None,
                                           x_steps=x_steps_DS_glass_rot,
                                           y_steps=y_steps_DS_glass_rot,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_synth                  = cfg_geom('cyl', z_steps_DS_glass_test, r_steps_800mm[0:],
                                           phi_steps_8[0:], x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_hel_true               = cfg_geom('cyl', z_steps_DS_glass_test, r_steps_800mm_true,
                                           phi_steps_true, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=True)

cfg_geom_hel_250mm              = cfg_geom('cyl', z_steps_DS_glass_small, r_steps_250mm[0:],
                                           phi_steps_8[0:], x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_hel_250mm2              = cfg_geom('cyl', z_steps_DS_glass_small, r_steps_250mm_true[0:],
                                            phi_steps_true[0:], x_steps=None, y_steps=None,
                                            bad_calibration=[False, False, False],
                                            interpolate=False, do2pi=True)

cfg_geom_hel_nubs               = cfg_geom('cyl', z_steps_DS_glass_small, r_steps_800mm_sml,
                                           phi_steps_8, x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

# cfg_geom_set_cyl_800mm_interp   = [
#     cfg_geom('cyl', z_steps_DS_interp, r_steps_interp,
#              phi_steps_interp, xy_steps=None,
#              bad_calibration=[True, False, False],
#              interpolate=1)
#     for i in range(syst_set)]

cfg_geom_set_cyl_800mm   = [
    cfg_geom('cyl', z_steps_DS_long, r_steps_800mm,
             phi_steps_8, x_steps=None, y_steps=None,
             bad_calibration=[True, False, False, i],
             interpolate=False, do2pi=False)
    for i in range(syst_set)]


###################
# the params cfgs #
###################
cfg_params_Mau_DS_opt                 = cfg_params(ns=3, ms=40, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=1)
cfg_params_Mau_DS_825mm_v1            = cfg_params(ns=3, ms=80, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=1)
cfg_params_Mau_DS_825mm_v2            = cfg_params(ns=3, ms=80, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=1)
cfg_params_Mau_DS_800mm               = cfg_params(ns=3, ms=50, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=5)
cfg_params_Mau_DS_800mm_bessel        = cfg_params(ns=4, ms=50, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=2)
cfg_params_Mau_DS_800mm_bessel_hybrid = cfg_params(ns=4, ms=50, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=3)
cfg_params_Mau_DS_700                 = cfg_params(ns=3, ms=70, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=1)
cfg_params_Mau_DS_bad                 = cfg_params(ns=3, ms=80, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=1)
cfg_params_GA05_DS_offset             = cfg_params(ns=4, ms=50, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=5)
cfg_params_Mau10_DS_offset            = cfg_params(ns=3, ms=50, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=1)
cfg_params_Mau_PS_opt                 = cfg_params(ns=3, ms=40, cns=0, cms=0, nms=0, Reff=9000,
                                                   n_scale=1, m_scale=1, func_version=1)

cfg_params_GA05_DS_seg_trk_no_mod     = cfg_params(ns=10, ms=35, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=5)

cfg_params_GA05_DS_seg_trk_mod1       = cfg_params(ns=6, ms=35, cns=10, cms=15, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=8)

cfg_params_GA05_DS_seg_trk_mod2       = cfg_params(ns=6, ms=35, cns=10, cms=15, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=9)


###################
# the pickle cfgs #
###################
cfg_pickle_GA05_seg_trk_no_mod     = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='GA05_seg_trk_no_mod',
                                                save_name='GA05_seg_trk_no_mod', recreate=False)

cfg_pickle_GA05_seg_trk_mod1       = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='GA05_seg_trk_mod1',
                                                save_name='GA05_seg_trk_mod1', recreate=False)

cfg_pickle_GA05_seg_trk_mod2       = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='GA05_seg_trk_mod2',
                                                save_name='GA05_seg_trk_mod2', recreate=True)

cfg_pickle_new_Mau                 = cfg_pickle(use_pickle=False, save_pickle=True, load_name=None,
                                                save_name='Mau10_opt', recreate=False)
cfg_pickle_Mau_700                 = cfg_pickle(use_pickle=True, save_pickle=False,
                                                load_name='Mau10_700', save_name='Mau10_700',
                                                recreate=True)

cfg_pickle_Mau_825mm_v1            = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_825mm_v1_tmp',
                                                save_name='Mau10_825mm_v1_tmp', recreate=False)
cfg_pickle_Mau_825mm_v2            = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_825mm_v2',
                                                save_name='Mau10_825mm_v2', recreate=True)

cfg_pickle_GA05_offset             = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='GA05_offset', save_name='GA05_offset',
                                                recreate=False)

cfg_pickle_Mau_800mm               = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='Mau10_800mm', save_name='Mau10_800mm',
                                                recreate=False)
cfg_pickle_Mau_800mm_bessel        = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='Mau10_800mm_bessel',
                                                save_name='Mau10_800mm_bessel', recreate=False)
cfg_pickle_Mau_800mm_bessel_hybrid = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='Mau10_800mm_bessel_hybrid',
                                                save_name='Mau10_800mm_bessel_hybrid',
                                                recreate=False)
cfg_pickle_Mau_800mm_20cm          = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_800mm_20cm',
                                                save_name='Mau10_800mm_20cm', recreate=True)
cfg_pickle_Mau_800mm_30cm          = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_800mm_30cm',
                                                save_name='Mau10_800mm_30cm', recreate=True)
cfg_pickle_Mau_800mm_40cm          = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_800mm_40cm',
                                                save_name='Mau10_800mm_40cm', recreate=True)
cfg_pickle_Mau_800mm_50cm          = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_800mm_50cm',
                                                save_name='Mau10_800mm_50cm', recreate=True)

cfg_pickle_Mau_bad_m_test_v1       = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_bad_m_test_v1',
                                                save_name='Mau10_bad_m_test_v1', recreate=True)
cfg_pickle_Mau_bad_p_test_v1       = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_bad_p_test_v1',
                                                save_name='Mau10_bad_p_test_v1', recreate=True)
cfg_pickle_Mau_bad_r_test_v1       = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='Mau10_bad_r_test_v1',
                                                save_name='Mau10_bad_r_test_v1', recreate=True)
cfg_pickle_Mau_bad_r2_test_v1      = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='Mau10_bad_r2_test_v1',
                                                save_name='Mau10_bad_r2_test_v1', recreate=True)

cfg_pickle_Mau_bad_m_test_v2       = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_bad_m_test_v2',
                                                save_name='Mau10_bad_m_test_v2', recreate=True)
cfg_pickle_Mau_bad_p_test_v2       = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_bad_p_test_v2',
                                                save_name='Mau10_bad_p_test_v2', recreate=True)
cfg_pickle_Mau_bad_r_test_v2       = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='Mau10_bad_r_test_v2',
                                                save_name='Mau10_bad_r_test_v2', recreate=True)

cfg_pickle_Mau_bad_m_test_req      = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_bad_m_test_req',
                                                save_name='Mau10_bad_m_test_req', recreate=True)
cfg_pickle_Mau_bad_p_test_req      = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_bad_p_test_req',
                                                save_name='Mau10_bad_p_test_req', recreate=True)
cfg_pickle_Mau_bad_r_test_req      = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_bad_r_test_req',
                                                save_name='Mau10_bad_r_test_req', recreate=True)

cfg_pickle_Mau_700_plotly          = cfg_pickle(use_pickle=True, save_pickle=False,
                                                load_name='Mau10_700', save_name='Mau10_700',
                                                recreate=True)

cfg_pickle_GA05_no_ext             = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='GA05_no_ext', save_name='GA05_no_ext',
                                                recreate=False)
cfg_pickle_GA05_no_DS              = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='GA05_no_DS', save_name='GA05_no_DS',
                                                recreate=False)

cfg_pickle_new_Mau_PS              = cfg_pickle(use_pickle=True, save_pickle=False,
                                                load_name='Mau10_PS', save_name='Mau10_PS',
                                                recreate=True)
cfg_pickle_new_Mau_PS_plotly       = cfg_pickle(use_pickle=True, save_pickle=False,
                                                load_name='Mau10_PS', save_name='Mau10_PS',
                                                recreate=True)

cfg_pickle_Mau_fullsim             = cfg_pickle(use_pickle=True, save_pickle=False,
                                                load_name='Mau10_700_test', save_name=None,
                                                recreate=True)

cfg_pickle_Mau_interp              = cfg_pickle(use_pickle=False, save_pickle=True,
                                                load_name='Mau10_interp', save_name='Mau10_interp',
                                                recreate=False)

cfg_pickle_GA02_800mm              = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='GA02_800mm', save_name='GA02_800mm',
                                                recreate=True)

cfg_pickle_Mau_800mm_interp        = cfg_pickle(use_pickle=True, save_pickle=True,
                                                load_name='Mau10_800mm_interp',
                                                save_name='Mau10_800mm_interp', recreate=True)

cfg_pickle_Mau_800mm_interp_v2      = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Mau10_800mm_interp_v2',
                                                 save_name='Mau10_800mm_interp_v2', recreate=False)

cfg_pickle_Mau_800mm_interp_v3      = cfg_pickle(use_pickle=True, save_pickle=True,
                                                 load_name='Mau10_800mm_interp_v3',
                                                 save_name='Mau10_800mm_interp_v3', recreate=False)

cfg_params_Glass_DS_Cyl               = cfg_params(ns=1, ms=60, cns=0, cms=0, nms=0, Reff=7000,
                                                   n_scale=1, m_scale=1, func_version=5)
cfg_pickle_Glass_Cyl                = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cyl_Only',
                                                 save_name='Cyl_Only', recreate=False)

cfg_params_Glass_DS_Bus             = cfg_params(ns=10, ms=10, cns=0, cms=0, nms=0, Reff=5000,
                                                 n_scale=1, m_scale=1, func_version=20)
cfg_pickle_Glass_Bus                = cfg_pickle(use_pickle=True, save_pickle=True,
                                                 load_name='Bus_Only_long',
                                                 save_name='Bus_Only_long', recreate=True)

cfg_params_Glass_DS_Combo           = cfg_params(ns=1, ms=60, cns=10, cms=10, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=6)
cfg_pickle_Glass_Combo              = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Combo',
                                                 save_name='Combo', recreate=True)

cfg_params_Glass_DS_Cyl_bad         = cfg_params(ns=6, ms=50, cns=0, cms=0, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=5)
cfg_pickle_Glass_Combo_Cyl          = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Combo_Cyl',
                                                 save_name='Combo_Cyl', recreate=False)

cfg_params_Glass_DS_Bus_Rot         = cfg_params(ns=20, ms=5, cns=0, cms=0, nms=0, Reff=5000,
                                                 n_scale=1, m_scale=1, func_version=33)
cfg_pickle_Glass_Bus_Rot            = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Bus_Rot',
                                                 save_name='Bus_Rot', recreate=False)

cfg_params_GA05_DS_800mm            = cfg_params(ns=2, ms=60, cns=10, cms=10, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=6)
cfg_pickle_GA05_800mm               = cfg_pickle(use_pickle=True, save_pickle=True,
                                                 load_name='GA05_BusTest', save_name='GA05_BusTest',
                                                 recreate=True)

cfg_params_GA05_DS_HelTest          = cfg_params(ns=15, ms=15, cns=10, cms=10, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=100)
cfg_pickle_GA05_HelTest             = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='GA05_HelTest', save_name='GA05_HelTest',
                                                 recreate=False)

cfg_params_Mau_DS_800mm_long        = cfg_params(ns=2, ms=100, cns=0, cms=0, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=5)
cfg_pickle_Mau_800mm_long           = cfg_pickle(use_pickle=True, save_pickle=True,
                                                 load_name='Mau10_800mm_long',
                                                 save_name='Mau10_800mm_long', recreate=False)

cfg_params_Glass_DS_Hel             = cfg_params(ns=7, ms=30, cns=0, cms=0, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=5)
cfg_pickle_Glass_Hel                = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Glass_Hel',
                                                 save_name='Glass_Hel', recreate=True)

cfg_params_Glass_DS_Hel_Pel         = cfg_params(ns=15, ms=15, cns=0, cms=0, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=110)
cfg_pickle_Glass_Hel_Pel            = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Glass_Hel_Pel',
                                                 save_name='Glass_Hel_Pel', recreate=False)

cfg_params_Glass_DS_Hel_Test        = cfg_params(ns=8, ms=8, cns=0, cms=0, nms=0, Reff=13350,
                                                 n_scale=1, m_scale=1, func_version=115)
cfg_pickle_Glass_Hel_Test           = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Glass_Hel_Test',
                                                 save_name='Glass_Hel_Test', recreate=False)
# load_name='Glass_Hel_Mau10',
# save_name='Glass_Hel_Mau10', recreate=False)

cfg_params_Glass_DS_Cyl_Test        = cfg_params(ns=3, ms=30, cns=5, cms=5, nms=0, Reff=15000,
                                                 n_scale=1, m_scale=1, func_version=7)
cfg_pickle_Glass_Cyl_Test           = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Glass_Cyl_Test',
                                                 save_name='Glass_Cyl_Test', recreate=False)

cfg_params_synth                    = cfg_params(ns=10, ms=10, cns=0, cms=0, nms=0, Reff=5000,
                                                 n_scale=1, m_scale=1, func_version=114)
cfg_pickle_synth                    = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='synth',
                                                 save_name='synth', recreate=False)

cfg_params_Mau13_DS                 = cfg_params(ns=5, ms=60, cns=0, cms=0, nms=0, Reff=7000,
                                                 n_scale=1, m_scale=1, func_version=6)
cfg_pickle_Mau13                    = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Mau13',
                                                 save_name='Mau13', recreate=False)

cfg_params_Cole_DS_Hel              = cfg_params(ns=10, ms=4, cns=0, cms=0, nms=0, Reff=13350,
                                                 n_scale=1, m_scale=1, func_version=119)
cfg_pickle_Cole_Hel                 = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cole_Hel',
                                                 save_name='Cole_Hel', recreate=False)

cfg_params_Cole_DS_Mau13            = cfg_params(ns=20, ms=2, cns=0, cms=0, nms=0, Reff=13350,
                                                 n_scale=1, m_scale=1, func_version=119)
cfg_pickle_Cole_Mau13               = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cole_Hel',
                                                 save_name='Cole_Hel', recreate=False)

cfg_params_Cole_DS_Hel250           = cfg_params(ns=1, ms=2, cns=0, cms=0, nms=0, Reff=0.1,
                                                 n_scale=100, m_scale=1, func_version=120)
cfg_pickle_Cole_Hel250              = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cole_Hel250',
                                                 save_name='Cole_Hel250', recreate=False)

cfg_params_Cole_DS_Nubs             = cfg_params(ns=0, ms=0, cns=0, cms=0, nms=0, Reff=0.5,
                                                 n_scale=9, m_scale=1, func_version=119)
cfg_pickle_Cole_Nubs                = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cole_Nubs',
                                                 save_name='Cole_Nubs', recreate=False)
nms = list(zip(range(1, 31), range(1, 31)))
nms += list(zip(range(1, 31), range(2, 32)))
nms += list(zip(range(1, 31), range(3, 33)))
nms += list(zip(range(1, 31), range(10, 40)))
nms += list(zip(range(4, 31), range(14, 41)))
nms += list(zip(range(6, 31), range(17, 42)))
nms += list(zip(range(7, 31), range(19, 43)))
nms += list(zip(range(8, 31), range(20, 44)))
nms += list(zip(range(9, 31), range(21, 45)))
cfg_params_Glass_DS_Hel_New         = cfg_params(ns=0, ms=0, cns=0, cms=0, nms=nms, Reff=25000,
                                                 n_scale=1, m_scale=1, func_version=111)
cfg_pickle_Glass_Hel_New            = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Glass_Hel_New',
                                                 save_name='Glass_Hel_New', recreate=False)
# cfg_pickle_set_Mau_bad_m            = [
#    cfg_pickle(use_pickle=True, save_pickle=True,
#               load_name='Mau10_800mm_interp',
#               save_name='Mau10_bad_m_{}'.format(i), recreate=False)
#    for i in range(syst_set)]

cfg_pickle_set_Mau_bad_m            = [
    cfg_pickle(use_pickle=True, save_pickle=True,
               load_name='Mau10_800mm_long',
               save_name='Mau10_bad_m_{}'.format(i), recreate=False)
    for i in range(syst_set)]

#################
# the plot cfgs #
#################
cfg_plot_none = cfg_plot('none', [-2, 2], 'html', None)
cfg_plot_mpl = cfg_plot('mpl', [-2, 2], 'html', None)
cfg_plot_mpl_high_lim = cfg_plot('mpl', [-10, 10], 'html', None)
cfg_plot_plotly_img = cfg_plot('plotly_html_img', [-2, 2], 'html', None)
cfg_plot_plotly_html = cfg_plot('plotly_html', [-2, 2], 'html', None)
cfg_plot_plotly_high_lim = cfg_plot('plotly', [-10, 10], 'html', None)


if __name__ == "__main__":
    import sys

    # do PS stuff
    # field_map_analysis('halltoy_150mm', cfg_data_PS_Mau10, cfg_geom_cyl_150mm,
    #                    cfg_params_Mau_PS_opt, cfg_pickle_new_Mau_PS, cfg_plot_mpl)
    # field_map_analysis('halltoy_150mm', cfg_data_PS_Mau10, cfg_geom_cyl_150mm,
    #                    cfg_params_Mau_PS_opt, cfg_pickle_new_Mau_PS_plotly, cfg_plot_plotly)

    # field_map_analysis('halltoy_825mm_v1_test', cfg_data_DS_Mau10, cfg_geom_cyl_825mm_v1,
    #                    cfg_params_Mau_DS_825mm_v1, cfg_pickle_Mau_825mm_v1, cfg_plot_mpl)
    # field_map_analysis('halltoy_825mm_v2_test', cfg_data_DS_Mau10, cfg_geom_cyl_825mm_v2,
    #                    cfg_params_Mau_DS_825mm_v2, cfg_pickle_Mau_825mm_v2, cfg_plot_mpl)
    # field_map_analysis('halltoy_825mm_v1_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc_test, cfg_params_Mau_DS_825mm_v1,
    #                    cfg_pickle_Mau_825mm_v1, cfg_plot_mpl)
    # field_map_analysis('halltoy_825mm_v2_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim,
    #                    cfg_params_Mau_DS_825mm_v2, cfg_pickle_Mau_825mm_v2, cfg_plot_mpl)
    # field_map_analysis('halltoy_825mm_v1_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim,
    #                    cfg_params_Mau_DS_825mm_v1, cfg_pickle_Mau_825mm_v1, cfg_plot_plotly)
    # field_map_analysis('halltoy_825mm_v2_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim,
    #                    cfg_params_Mau_DS_825mm_v2, cfg_pickle_Mau_825mm_v2, cfg_plot_plotly)

    # Doing Z-spacing analyses
    # field_map_analysis('halltoy_800mm_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim_trunc,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm, cfg_plot_plotly)
    # field_map_analysis('halltoy_800mm_20cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_20cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_20cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_20cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_20cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_20cm, cfg_plot_plotly)
    # field_map_analysis('halltoy_800mm_20cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_20cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_20cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_20cm, cfg_plot_plotly)

    # field_map_analysis('halltoy_800mm_30cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_30cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_30cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_30cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_30cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_30cm, cfg_plot_plotly)
    # field_map_analysis('halltoy_800mm_30cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_30cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_30cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_30cm, cfg_plot_plotly)

    # field_map_analysis('halltoy_800mm_40cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_40cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_40cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_40cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_40cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_40cm, cfg_plot_plotly)
    # field_map_analysis('halltoy_800mm_40cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_40cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_40cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_40cm, cfg_plot_plotly)

    # field_map_analysis('halltoy_800mm_50cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_50cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_50cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_50cm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_50cm,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm_50cm, cfg_plot_plotly)
    # field_map_analysis('halltoy_800mm_50cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_50cm, cfg_plot_mpl)
    # field_map_analysis('halltoy_800mm_50cm_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_800mm,
    #                    cfg_pickle_Mau_800mm_50cm, cfg_plot_plotly)

    # Mis-calibration studies
    # field_map_analysis('halltoy_825mm_v1_bad_measure', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_bad_measure_v1, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_m_test_v1, cfg_plot_mpl_high_lim)
    # field_map_analysis('halltoy_825mm_v1_bad_position', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_bad_position_v1, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_p_test_v1, cfg_plot_plotly_high_lim)
    # field_map_analysis('halltoy_825mm_v1_bad_rotation', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_bad_rotation_v1, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_r_test_v1, cfg_plot_mpl_high_lim)
    # field_map_analysis('halltoy_825mm_v1_bad_rotation2', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_bad_rotation2_v1, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_r2_test_v1, cfg_plot_mpl_high_lim)

    # field_map_analysis('halltoy_825mm_v2_bad_measure', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_bad_measure_v2, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_m_test_v2, cfg_plot_plotly_high_lim)
    # field_map_analysis('halltoy_825mm_v2_bad_position', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_bad_position_v2, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_p_test_v2, cfg_plot_plotly_high_lim)
    # field_map_analysis('halltoy_825mm_v2_bad_rotation', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_bad_rotation_v2, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_r_test_v2, cfg_plot_plotly_high_lim)

    # field_map_analysis('halltoy_825mm_v1_bad_measure_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_m_test_v1, cfg_plot_mpl_high_lim)
    # field_map_analysis('halltoy_825mm_v1_bad_position_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_p_test_v1, cfg_plot_mpl_high_lim)
    # field_map_analysis('halltoy_825mm_v1_bad_rotation_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_r_test_v1, cfg_plot_mpl_high_lim)
    # field_map_analysis('halltoy_825mm_v1_bad_rotation_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v1,
    #                    cfg_plot_plotly_high_lim)

    # field_map_analysis('halltoy_825mm_v1_bad_rotation_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad,
    #                    cfg_pickle_Mau_bad_r2_test_v1, cfg_plot_mpl_high_lim)

    # field_map_analysis('halltoy_825mm_v2_bad_measure_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_m_test_v2,
    #                    cfg_plot_plotly_high_lim)
    # field_map_analysis('halltoy_825mm_v2_bad_position_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_p_test_v2,
    #                    cfg_plot_plotly_high_lim)
    # field_map_analysis('halltoy_825mm_v2_bad_rotation_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v2,
    #                    cfg_plot_mpl_high_lim)
    # field_map_analysis('halltoy_825mm_v2_bad_rotation_fullsim', cfg_data_DS_Mau10,
    #                    cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v2,
    #                    cfg_plot_plotly_high_lim)

    # field_map_analysis('halltoy_GA05_no_ext', cfg_data_DS_GA05_no_ext, cfg_geom_cyl_700mm,
    #                    cfg_params_GA05_DS_opt, cfg_pickle_GA05_no_ext, cfg_plot_mpl)
    # field_map_analysis('halltoy_GA05_no_DS', cfg_data_DS_GA05_no_DS, cfg_geom_cyl_700mm,
    #                    cfg_params_GA05_DS_opt, cfg_pickle_GA05_no_DS, cfg_plot_mpl)
    # field_map_analysis('halltoy_GA05_offset', cfg_data_DS_GA05_offset, cfg_geom_cyl_offset,
    #                    cfg_params_GA05_DS_offset, cfg_pickle_GA05_offset, cfg_plot_mpl)
    # field_map_analysis('halltoy_Mau10_offset', cfg_data_DS_Mau10_offset, cfg_geom_cyl_offset,
    #                    cfg_params_GA05_DS_800mm, cfg_pickle_GA05_offset, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm', cfg_data_DS_Mau10, cfg_geom_cyl_800mm,
    #                              cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long', cfg_data_DS_Mau10_long,
    #                              cfg_geom_cyl_800mm_long, cfg_params_Mau_DS_800mm_long,
    #                              cfg_pickle_Mau_800mm_long, cfg_plot_mpl)
    # make_csv(ff.params.valuesdict())

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_interp', cfg_data_DS_Mau10_long,
    #                             cfg_geom_cyl_800mm_interp, cfg_params_Mau_DS_800mm_long,
    #                             # cfg_geom_cyl_fullsim800, cfg_params_Mau_DS_800mm_long,
    #                             cfg_pickle_Mau_800mm_interp, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_interp_v2', cfg_data_DS_Mau10_long,
    #                              cfg_geom_cyl_800mm_interp_v2, cfg_params_Mau_DS_800mm_long,
    #                              cfg_pickle_Mau_800mm_interp_v2, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_interp_v3', cfg_data_DS_Mau10_long,
    #                              cfg_geom_cyl_800mm_interp_v3, cfg_params_Mau_DS_800mm_long,
    #                              cfg_pickle_Mau_800mm_interp_v3, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long_fullsim800', cfg_data_DS_Mau10_long,
    #                              cfg_geom_cyl_fullsim800, cfg_params_Mau_DS_800mm_long,
    #                              cfg_pickle_Mau_800mm_long, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_GA02_800mm', cfg_data_DS_GA02, cfg_geom_cyl_800mm_long,
    #                              cfg_params_Mau_DS_800mm_long, cfg_pickle_GA02_800mm,
    #                              cfg_plot_plotly)

    # hmd, ff = field_map_analysis('halltoy_GA05_grad', cfg_data_DS_GA05_grad,
    #                              cfg_geom_cyl_800mm_grad, cfg_params_GA05_DS_800mm,
    #                              cfg_pickle_GA05_800mm, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_GA05_800mm', cfg_data_DS_GA05, cfg_geom_cyl_800mm_long,
    #                              cfg_params_GA05_DS_800mm, cfg_pickle_GA05_800mm,
    #                              cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_GA05_seg_trk_no_mod', cfg_data_DS_GA05_seg_trk,
    #                              cfg_geom_cyl_seg_trk, cfg_params_GA05_DS_seg_trk_no_mod,
    #                              cfg_pickle_GA05_seg_trk_no_mod, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_GA05_seg_trk_mod1', cfg_data_DS_GA05_seg_trk2,
    #                              cfg_geom_cyl_seg_trk2, cfg_params_GA05_DS_seg_trk_mod1,
    #                              cfg_pickle_GA05_seg_trk_mod1, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_GA05_seg_trk_mod2', cfg_data_DS_GA05_seg_trk,
    #                              cfg_geom_cyl_seg_trk, cfg_params_GA05_DS_seg_trk_mod2,
    #                              cfg_pickle_GA05_seg_trk_mod2, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long_bad_m_req', cfg_data_DS_Mau10_long,
    #                             cfg_geom_cyl_bad_measure_req, cfg_params_Mau_DS_800mm_long,
    #                              cfg_pickle_Mau_bad_m_test_req, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long_bad_p_req', cfg_data_DS_Mau10_long,
    #                              cfg_geom_cyl_bad_position_req, cfg_params_Mau_DS_800mm_long,
    #                              cfg_pickle_Mau_bad_p_test_req, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long_bad_r_req', cfg_data_DS_Mau10_long,
    #                              cfg_geom_cyl_bad_rotation_req, cfg_params_Mau_DS_800mm_long,
    #                              cfg_pickle_Mau_bad_r_test_req, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long_bad_m_req_fullsim',
    #                              cfg_data_DS_Mau10_long, cfg_geom_cyl_fullsim800,
    #                              cfg_params_Mau_DS_800mm_long, cfg_pickle_Mau_bad_m_test_req,
    #                              cfg_plot_mpl)
    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long_bad_p_req_fullsim',
    #                              cfg_data_DS_Mau10_long, cfg_geom_cyl_fullsim800,
    #                              cfg_params_Mau_DS_800mm_long, cfg_pickle_Mau_bad_p_test_req,
    #                              cfg_plot_mpl)
    # hmd, ff = field_map_analysis('halltoy_Mau10_800mm_long_bad_r_req_fullsim',
    #                              cfg_data_DS_Mau10_long, cfg_geom_cyl_fullsim800,
    #                              cfg_params_Mau_DS_800mm_long, cfg_pickle_Mau_bad_r_test_req,
    #                              cfg_plot_mpl)

    # field_map_analysis('halltoy_interp', cfg_data_DS_Mau10, cfg_geom_cyl_800mm_interp_slice,
    #                    cfg_params_Mau_DS_800mm, cfg_pickle_Mau_interp, cfg_plot_mpl)

    # for i in range(syst_set):
    #     hmd, ff = field_map_analysis('halltoy_Mau10_800mm_interp_bad_m_{}'.format(i),
    #                                  cfg_data_DS_Mau10_long,
    #                               cfg_geom_set_cyl_800mm_interp[i], cfg_params_Mau_DS_800mm_long,
    #                                  cfg_pickle_set_Mau_bad_m[i], cfg_plot_mpl)

    # for i in range(syst_set):
    #     field_map_analysis('halltoy_Mau10_800mm_bad_m_{}'.format(i),
    #                        cfg_data_DS_Mau10_long,
    #                        cfg_geom_set_cyl_800mm[i], cfg_params_Mau_DS_800mm_long,
    #                        cfg_pickle_set_Mau_bad_m[i], cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Cyl_Only', cfg_data_DS_Glass_Cyl,
    #                              cfg_geom_cyl_glass, cfg_params_Glass_DS_Cyl,
    #                              cfg_pickle_Glass_Cyl, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Bus_Only', cfg_data_DS_Glass_Bus,
    #                              cfg_geom_bus_glass, cfg_params_Glass_DS_Bus,
    #                              cfg_pickle_Glass_Bus, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Combo_cyl_only_fit', cfg_data_DS_Glass_Combo,
    #                              cfg_geom_cyl_glass, cfg_params_Glass_DS_Cyl_bad,
    #                              cfg_pickle_Glass_Combo_Cyl, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Combo', cfg_data_DS_Glass_Combo,
    #                              cfg_geom_cyl_glass, cfg_params_Glass_DS_Combo,
    #                              cfg_pickle_Glass_Combo, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Bus_Rot', cfg_data_DS_Glass_Bus_Rot,
    #                              cfg_geom_bus_rot_glass, cfg_params_Glass_DS_Bus_Rot,
    #                              cfg_pickle_Glass_Bus_Rot, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Hel', cfg_data_DS_Glass_Hel,
    #                              cfg_geom_hel_glass, cfg_params_Glass_DS_Hel,
    #                              cfg_pickle_Glass_Hel, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Hel_Pel', cfg_data_DS_Glass_Hel,
    #                              cfg_geom_hel_glass, cfg_params_Glass_DS_Hel_Pel,
    #                              cfg_pickle_Glass_Hel_Pel, cfg_plot_plotly_html)

    # hmd, ff = field_map_analysis('halltoy_Glass_Hel_Pel_4', cfg_data_DS_Glass_Hel,
    #                              cfg_geom_hel_glass, cfg_params_Glass_DS_Hel_Pel,
    #                              cfg_pickle_Glass_Hel_Pel, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Hel_Test', cfg_data_DS_Glass_Hel_3,
    #                              cfg_geom_hel_glass_test, cfg_params_Glass_DS_Hel_Test,
    #                              cfg_pickle_Glass_Hel_Test, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Hel_Test', cfg_data_DS_Glass_Hel_4,
    #                              cfg_geom_hel_glass_test, cfg_params_Glass_DS_Hel_Test,
    #                              cfg_pickle_Glass_Hel_Test, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Hel_Small', cfg_data_DS_Glass_Hel_3S,
    #                              cfg_geom_hel_glass_small, cfg_params_Glass_DS_Hel_Test,
    #                              cfg_pickle_Glass_Hel_Test, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Cyl_Test', cfg_data_DS_Glass_Hel_3,
    #                              cfg_geom_hel_glass_test, cfg_params_Glass_DS_Cyl_Test,
    #                              cfg_pickle_Glass_Cyl_Test, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_synth', cfg_data_synth,
    #                              cfg_geom_synth, cfg_params_synth,
    #                              cfg_pickle_synth, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau10_Pel', cfg_data_DS_Mau10, cfg_geom_cyl_800mm,
    #                              cfg_params_Glass_DS_Hel_Test, cfg_pickle_Glass_Hel_Test,
    #                              cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Glass_Hel_New', cfg_data_DS_Glass_Hel_3,
    #                              cfg_geom_hel_glass_test, cfg_params_Glass_DS_Hel_New,
    #                              cfg_pickle_Glass_Hel_New, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_GA05_HelTest', cfg_data_DS_GA05,
    #                              cfg_geom_cyl_800mm_long, cfg_params_GA05_DS_HelTest,
    #                              cfg_pickle_GA05_HelTest, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau13', cfg_data_DS_Mau13,
    #                              cfg_geom_cyl_800mm_long, cfg_params_Mau13_DS,
    #                              cfg_pickle_Mau13, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau13', cfg_data_DS_Mau13,
    #                              cfg_geom_cyl_fullsim_trunc, cfg_params_Mau13_DS,
    #                              cfg_pickle_Mau13, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Cole_Hel', cfg_data_DS_Cole_Hel_true,
    #                              cfg_geom_hel_true, cfg_params_Cole_DS_Hel,
    #                              cfg_pickle_Cole_Hel, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Cole_Hel250', cfg_data_DS_Cole_Hel_10x,
    #                              cfg_geom_hel_250mm, cfg_params_Cole_DS_Hel250,
    #                              cfg_pickle_Cole_Hel250, cfg_plot_mpl)

    hmd, ff = field_map_analysis('halltoy_Cole_Hel250', cfg_data_DS_Cole_Hel_10x2,
                                 cfg_geom_hel_250mm2, cfg_params_Cole_DS_Hel250,
                                 cfg_pickle_Cole_Hel250, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Cole_Nubs', cfg_data_DS_Cole_Nubs_descale,
    #                              cfg_geom_hel_nubs, cfg_params_Cole_DS_Nubs,
    #                              cfg_pickle_Cole_Nubs, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('halltoy_Mau13', cfg_data_DS_Mau13,
    #                              cfg_geom_cyl_800mm_trk, cfg_params_Cole_DS_Mau13,
    #                              cfg_pickle_Cole_Mau13, cfg_plot_mpl)
