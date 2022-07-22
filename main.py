from doctest import OutputChecker
from PLS import *
from PCR import *
from LINEAR_REG import *
from utilities import *
import pandas as pd
import numpy as np
import os
import sys

DEBUG = False
ktop_1D = int(sys.argv[1])
ktop_2D = int(sys.argv[2])
ADJUSTED_R2 = True if sys.argv[3] == "-adjusted" else False
KTOP_METHOD = sys.argv[4]# "other" #"MtM"
n_component_max = int(sys.argv[5])
method = sys.argv[6]

# phenotype_list = ["P31", "As75", "Cd114", "Cu65", "Li7", "Mn55", "Na23", "S34", "Sr88", "Ca43", "Co59", "Fe57", "K39", "Mg25", "Rb85", "Se82", "Zn66", "Mo98"]
# data_set_list = ['Arabidopsis_data_set/376K']
phenotype_list = ["tgres", "hdlres", "ldlres", "bmires", "crpres", "glures", "insres", "sysres", "diares"]
data_set_list = ['NFBC1966_Human']
# phenotype_list = ["31", "1", "75"]
# data_set_list = ['Atwell_et_al_data_set']
nth_elem_list = {"31":'169157', "1":'171400', "75":'165448'}
for data_set in data_set_list:
    if data_set=='Arabidopsis_data_set/161K':
        X_matrix_path = "/data/home/carluerjb/Documents/data/data_for_NEO/X/X_no_header_norm/X161KNH_norm.npy"
    if data_set=='Arabidopsis_data_set/376K':
        X_matrix_path = "/data/home/carluerjb/Documents/data/data_for_NEO/X/X_no_header_norm/X346KNH_norm.npy"
    if data_set=='NFBC1966_Human':
        X_matrix_path = "/data/home/carluerjb/Documents/data/data_for_NEO/X_human/X_no_header_no_NaN_no_ID_norm/X_human_full_no_NaN_norm.npy"
    if data_set=='Atwell_et_al_data_set':
        X_matrix_path_part = "/data/home/carluerjb/Documents/data/data_for_NEO/X_Atwell/04_06/no_header_norm/"
    else:
        if X_matrix_path.split('.')[1]=="npy":
            X = np.load(X_matrix_path)
            X = pd.DataFrame(X)
        else:
            X = pd.read_table(X_matrix_path, sep=' ', header=None)
        X_center = X.sub(X.mean())
        X = X_center.div(X.std())
        nb_SNP = len(X.columns)
        nb_indiv = len(X.index)
        assert (ktop_1D<nb_indiv)==(ktop_2D<nb_indiv), "ktop1D < nb individuals and ktop2D > nb individuals : Can't compare mixed methods !"

    for phenotype in phenotype_list:
        if data_set=='Atwell_et_al_data_set':
            X_matrix_path = X_matrix_path_part + phenotype + ".npy"
            if X_matrix_path.split('.')[1]=="npy":
                X = np.load(X_matrix_path)
                X = pd.DataFrame(X)
            else:
                X = pd.read_table(X_matrix_path, sep=' ', header=None)
            X_center = X.sub(X.mean())
            X = X_center.div(X.std())
            nb_SNP = len(X.columns)
            nb_indiv = len(X.index)
            assert (ktop_1D<nb_indiv)==(ktop_2D<nb_indiv), "ktop1D < nb individuals and ktop2D > nb individuals : Can't compare mixed methods !"
        out_path = "/data/home/carluerjb/Documents/data/data_for_NEO/PCR_R2/" + data_set + "/" + phenotype + "/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if data_set=="NFBC1966_Human":
            Y_vector_path = "/data/home/carluerjb/Documents/data/data_for_NEO/Y_human/Y_norm/" + phenotype + ".txt"
        elif data_set=='Atwell_et_al_data_set':
            Y_vector_path = "/data/home/carluerjb/Documents/data/data_for_NEO/Y_Atwell/Y_HDF5/Y_norm/" + phenotype + ".txt"
        else:
            Y_vector_path = "/data/home/carluerjb/Documents/data/data_for_NEO/Y/" + phenotype + ".txt"
        Y = np.asarray(np.loadtxt(Y_vector_path))
        Y = normalise(Y)
        print(Y)
        print("Working on phenotype : ", phenotype)
        if data_set=="Arabidopsis_data_set/161K":
            theta_1D_path = "/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out/" + phenotype + "_Ynorm/diag"
            theta_2D_path = "/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out/" + phenotype + "_Ynorm/161152"
        elif data_set=="Arabidopsis_data_set/376K":
            theta_1D_path = "/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out_346K/" + phenotype + "_Ynorm/diag"
            theta_2D_path = "/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out_346K/" + phenotype + "_Ynorm/346094"
        elif data_set=="NFBC1966_Human":
            theta_1D_path = "/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out_NFBC1966_Human/" + phenotype + "/diag"
            theta_2D_path = "/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out_NFBC1966_Human/" + phenotype + "/331476"
        elif data_set=='Atwell_et_al_data_set':
            theta_1D_path = "/data/home/carluerjb/Documents/data/data_for_NEO//nth_element_out_Atwell/" + phenotype + "/diag"
            theta_2D_path = "/data/home/carluerjb/Documents/data/data_for_NEO//nth_element_out_Atwell/" + phenotype + "/" + nth_elem_list[phenotype]
        else:
            exit(1)

        # STD MODE
        Theta_ID = np.asarray(np.loadtxt(theta_1D_path + ".ktop", dtype=int))
        Theta_hat_ID = pd.read_table(theta_2D_path + ".ktop", header=None, names=['K'], dtype={'K' : int})
        Theta = np.asarray(np.loadtxt(theta_1D_path + ".kval"))
        Theta_hat = pd.read_table(theta_2D_path + ".kval", header=None)[0].to_numpy()
        X_gen, Xp, Z, Theta, Theta_hat, Theta_ID, Theta_hat_ID = generate_X_Xp_Z_Thetas(X, Theta_hat, Theta_hat_ID, Theta, Theta_ID, ktop1D=ktop_1D, ktop2D=ktop_2D, ktop_method="BF")
        # if DEBUG!=True:
        #     if KTOP_METHOD=="MtM":
        #         pass
        #     else:
        #         run_and_save(
        #             computePCR, 
        #             out_path + "R2_PCR.txt", 
        #             Y, 
        #             datas=[X_gen, Xp], 
        #             data_label=["1D", "1D+2D"], 
        #             adjusted_r2=ADJUSTED_R2,
        #             n_component=n_component_max)
        if KTOP_METHOD=="MtM":
            # r2_score_1D_2D_pcr = run_and_save_multiY_plot_tt(computeManyLmPlot, 
            #                                                         Y, 
            #                                                         out = out_path,
            #                                                         study = "LM",
            #                                                         datas=Xp, 
            #                                                         title=["1D+2D"], 
            #                                                         adjusted_r2=ADJUSTED_R2,
            #                                                         n_component=n_component_max)
            r2_score_1D_2D_pcr = run_and_save_multiY_plot_tt(computeManyLmPlotMano, 
                                                                    Y, 
                                                                    out = out_path,
                                                                    study = "LM",
                                                                    datas=Xp, 
                                                                    title=["1D+2D"], 
                                                                    adjusted_r2=ADJUSTED_R2,
                                                                    n_component=n_component_max)
        else:
            if method == "PCR":
                r2_score_1D_pcr, r2_score_1D_2D_pcr = run_and_save_multiY_plot(computePCR_plot, 
                                                                        Y, 
                                                                        out = out_path,
                                                                        study = "PCR",
                                                                        datas=[X_gen, Xp], 
                                                                        title=["1D", "2D"], 
                                                                        adjusted_r2=ADJUSTED_R2,
                                                                        n_component=n_component_max)
            else:
                r2_score_1D_pcr, r2_score_1D_2D_pcr = run_and_save_multiY_plot(computePLS_plot, 
                                                                        Y, 
                                                                        out = out_path,
                                                                        study = "PLS",
                                                                        datas=[X_gen, Xp], 
                                                                        title=["1D", "2D"], 
                                                                        adjusted_r2=ADJUSTED_R2,
                                                                        n_component=n_component_max)
        r2_score_1D_random_2D_pcr_random_l = []
        r2_score_1D_pcr_random_l = []
        r2_score_1D_random_2D_pcr_random_l = []
        r2_score_1D_2D_pcr_random_l = []
        r2_score_1D_pcr_random_l = []
        r2_score_1D_2D_pcr_random_l = []
        for i in range(0,5):
            # 1D_random, 2D_random mode
            out_path = "/data/home/carluerjb/Documents/data/data_for_NEO/PCR_R2/" + data_set + "_Random/" + phenotype + "/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            Theta_ID = K_random_index_generator(nb_SNP=nb_SNP, nb_ktop=20000, inter=False)['K'].to_numpy()
            Theta_hat_ID = K_random_index_generator(nb_SNP=nb_SNP, nb_ktop=20000)
            Theta = np.asarray(np.loadtxt(theta_1D_path + ".kval"))
            Theta_hat = pd.read_table(theta_2D_path + ".kval", header=None)[0].to_numpy()
            X_gen, Xp, Z, Theta, Theta_hat, Theta_ID, Theta_hat_ID = generate_X_Xp_Z_Thetas(X, Theta_hat, Theta_hat_ID, Theta, Theta_ID, ktop1D=ktop_1D, ktop2D=ktop_2D, ktop_method="BF")
            # if DEBUG!=True:
            #     if KTOP_METHOD=="MtM":
            #         pass
            #     else:
            #         run_and_save(
            #             computePCR, 
            #             out_path + "R2_PCR.txt", 
            #             Y, 
            #             datas=[X_gen, Xp], 
            #             data_label=["1D", "1D+2D"], 
            #             adjusted_r2=ADJUSTED_R2, 
            #             n_component=n_component_max)
            if KTOP_METHOD=="MtM":
                # r2_score_1D_random_2D_pcr_random = run_and_save_multiY_plot_tt(computeManyLmPlot, 
                #                                                         Y, 
                #                                                         out = out_path,
                #                                                         study = "LM",
                #                                                         datas=Xp, 
                #                                                         title=["1D+2D"], 
                #                                                         adjusted_r2=ADJUSTED_R2)
                r2_score_1D_random_2D_pcr_random = run_and_save_multiY_plot_tt(computeManyLmPlotMano, 
                                                                        Y, 
                                                                        out = out_path,
                                                                        study = "LM",
                                                                        datas=Xp, 
                                                                        title=["1D+2D"], 
                                                                        adjusted_r2=ADJUSTED_R2)
                r2_score_1D_random_2D_pcr_random_l.append(r2_score_1D_random_2D_pcr_random)
            else:
                if method == "PCR":
                    r2_score_1D_pcr_random, r2_score_1D_random_2D_pcr_random = run_and_save_multiY_plot(computePCR_plot, 
                                                                                    Y, 
                                                                                    out = out_path,
                                                                                    study = "PCR",
                                                                                    datas=[X_gen, Xp], 
                                                                                    title=["1D", "2D"], 
                                                                                    adjusted_r2=ADJUSTED_R2, n_component=n_component_max)
                else:
                    r2_score_1D_pcr_random, r2_score_1D_random_2D_pcr_random = run_and_save_multiY_plot(computePLS_plot, 
                                                                                    Y, 
                                                                                    out = out_path,
                                                                                    study = "PLS",
                                                                                    datas=[X_gen, Xp], 
                                                                                    title=["1D", "2D"], 
                                                                                    adjusted_r2=ADJUSTED_R2, n_component=n_component_max)
                r2_score_1D_pcr_random_l.append(r2_score_1D_pcr_random)
                r2_score_1D_random_2D_pcr_random_l.append(r2_score_1D_random_2D_pcr_random)

            # 1D std, 2D random mode
            out_path = "/data/home/carluerjb/Documents/data/data_for_NEO/PCR_R2/" + data_set + "_Final/" + phenotype + "/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            Theta_ID = np.asarray(np.loadtxt(theta_1D_path + ".ktop", dtype=int))
            Theta_hat_ID = K_random_index_generator(nb_SNP=nb_SNP, nb_ktop=20000)
            Theta = np.asarray(np.loadtxt(theta_1D_path + ".kval"))
            Theta_hat = pd.read_table(theta_2D_path + ".kval", header=None)[0].to_numpy()
            X_gen, Xp, Z, Theta, Theta_hat, Theta_ID, Theta_hat_ID = generate_X_Xp_Z_Thetas(X, Theta_hat, Theta_hat_ID, Theta, Theta_ID, ktop1D=ktop_1D, ktop2D=ktop_2D, ktop_method="BF")
            # if DEBUG!=True:
            #     if KTOP_METHOD=="MtM":
            #         pass
            #     else:
            #         run_and_save(
            #             computePCR, 
            #             out_path + "R2_PCR.txt", 
            #             Y, 
            #             datas=[X_gen, Xp], 
            #             data_label=["1D", "1D+2D"], 
            #             adjusted_r2=ADJUSTED_R2, 
            #             n_component=n_component_max)
            
            if KTOP_METHOD=="MtM":
                # r2_score_1D_2D_pcr_random = run_and_save_multiY_plot_tt(computeManyLmPlot, 
                #                                                         Y, 
                #                                                         out = out_path,
                #                                                         study = "LM",
                #                                                         datas=Xp, 
                #                                                         title=["1D+2D"], 
                #                                                         adjusted_r2=ADJUSTED_R2)
                r2_score_1D_2D_pcr_random = run_and_save_multiY_plot_tt(computeManyLmPlotMano, 
                                                                        Y, 
                                                                        out = out_path,
                                                                        study = "LM",
                                                                        datas=Xp, 
                                                                        title=["1D+2D"], 
                                                                        adjusted_r2=ADJUSTED_R2)
                r2_score_1D_2D_pcr_random_l.append(r2_score_1D_2D_pcr_random)
            else:
                if method=="PCR":
                    r2_score_1D_pcr_random, r2_score_1D_2D_pcr_random = run_and_save_multiY_plot(computePCR_plot, 
                                                                                    Y, 
                                                                                    out = out_path,
                                                                                    study = "PCR",
                                                                                    datas=[X_gen, Xp], 
                                                                                    title=["1D", "2D"], 
                                                                                    adjusted_r2=ADJUSTED_R2, n_component=n_component_max)
                else:
                    r2_score_1D_pcr_random, r2_score_1D_2D_pcr_random = run_and_save_multiY_plot(computePLS_plot, 
                                                                                Y, 
                                                                                out = out_path,
                                                                                study = "PLS",
                                                                                datas=[X_gen, Xp], 
                                                                                title=["1D", "2D"], 
                                                                                adjusted_r2=ADJUSTED_R2, n_component=n_component_max)
                r2_score_1D_2D_pcr_random_l.append(r2_score_1D_2D_pcr_random)
                #r2_score_1D_pcr_random_l.append(r2_score_1D_pcr_random)
            

        if KTOP_METHOD=="MtM":
            plot_multiple_Y_plot_many_rand(list(range(0, len(r2_score_1D_2D_pcr)*1, 1)),
                                            y=[r2_score_1D_2D_pcr],
                                            y_rand=[r2_score_1D_2D_pcr_random_l, r2_score_1D_random_2D_pcr_random_l],
                                            outplot_path = out_path + "LM_multiplot.png",
                                            legend = ["1D+2D", "1D+2D_random", "1D_random+2D_random"],
                                            line_type=["solid", "dotted", "dashed"], debug=DEBUG, adjusted_r2=ADJUSTED_R2, study="LM"
                                            )
        else:
            if method=="PCR":
                print("HERE")
                plot_multiple_Y_plot_many_rand(list(range(1, (n_component_max+1))),
                                                y=[r2_score_1D_pcr, r2_score_1D_2D_pcr],
                                                y_rand=[r2_score_1D_pcr_random_l, r2_score_1D_2D_pcr_random_l, r2_score_1D_random_2D_pcr_random_l],
                                                outplot_path = out_path + "PCR_multiplot.png",
                                                legend = ["1D", "1D+2D", "1D_random", "1D+2D_random", "1D_random+2D_random"],
                                                line_type=["solid", "solid", "dotted", "dashed", "dashdot"], debug=DEBUG, adjusted_r2=ADJUSTED_R2, study="PCR"
                                                )
            else:
                plot_multiple_Y_plot_many_rand(list(range(1, (n_component_max+1))),
                                            y=[r2_score_1D_pcr, r2_score_1D_2D_pcr],
                                            y_rand=[r2_score_1D_pcr_random_l, r2_score_1D_2D_pcr_random_l, r2_score_1D_random_2D_pcr_random_l],
                                            outplot_path = out_path + "PLS_multiplot.png",
                                            legend = ["1D", "1D+2D", "1D_random", "1D+2D_random", "1D_random+2D_random"],
                                            line_type=["solid", "solid", "dotted", "dashed", "dashdot"], debug=DEBUG, adjusted_r2=ADJUSTED_R2, study="PLS"
                                            )

        if DEBUG!=True:
            if KTOP_METHOD=="MtM":
                save_results([r2_score_1D_2D_pcr, r2_score_1D_2D_pcr, r2_score_1D_2D_pcr_random], out_path + "LM.txt")
            else:
                save_results([r2_score_1D_pcr, r2_score_1D_2D_pcr, r2_score_1D_pcr_random_l, r2_score_1D_2D_pcr_random_l, r2_score_1D_random_2D_pcr_random_l], out_path + "pcr.txt")