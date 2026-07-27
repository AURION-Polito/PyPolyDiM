# _LICENSE_HEADER_
#
# Copyright (C) 2026.
# Terms register on the GPL-3.0 license.
#
# This file can be redistributed and/or modified under the license terms.
#
# See top level LICENSE file for more details.
#
# This file can be used citing references in CITATION.cff file.

from Brinkman_DF_PCC_2D.test_definition import StokesSinSin, Brinkman, ProblemType
from Brinkman_DF_PCC_2D.assembler import *
import os

def create_test(test_id: int) -> ITest:

    match test_id:
        case 1:
            return StokesSinSin()
        case 2:
            return Brinkman()
        case _:
            raise ValueError("not valid test id")


def create_mesh(geometry_utilities: gedim.GeometryUtilities, mesh_utilities: gedim.MeshUtilities,
                mesh_type: polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D,
                mesh_max_relative_area: float, import_path: str,
                pde_domain: polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D,
                mesh: gedim.MeshMatricesDAO):

    if (polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.triangular == mesh_type or
            polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.minimal == mesh_type or
            polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.polygonal == mesh_type or
            polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.squared == mesh_type or
        polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.random_distorted == mesh_type):

        polydim.pde_tools.mesh.pde_mesh_utilities.create_mesh_2_d(geometry_utilities,
                                                                  mesh_utilities,
                                                                  mesh_type,
                                                                  pde_domain,
                                                                  mesh_max_relative_area,
                                                                  mesh)

    elif (polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.csv_importer == mesh_type or
            polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.off_importer == mesh_type):

        polydim.pde_tools.mesh.pde_mesh_utilities.import_mesh_2_d(geometry_utilities,
                                                                  mesh_utilities,
                                                                  mesh_type,
                                                                  import_path,
                                                                  mesh)
    else:
        raise ValueError("MeshGenerator " + str(mesh_type) + " not supported")


def export_errors(file_path: str, test_id: int, mesh_type: int, method_id: int, method_order: int,
                  mesh:  gedim.MeshMatricesDAO,
                  count_do_fs_data: assembler_utilities.CountDOFsData,
                  post_process_data: PostProcessData,
                  file_separator = ';') -> None:

    print("{:<20} {:<15} {:<20} {:<5} {:<10} {:<10} {:<10} {:<10} {:<15} {:<15} {:<15} {:<15} {:<10}"
          .format('Test', 'Mesh', 'Method', 'Order', 'Cell2Ds', 'DOFs',
                  'Strong', 'h', 'errorL2Pressure',  'errorH1Velocity', 'normL2Pressure', 'normH1Velocity', 'residual'))

    print(
        "{:<20s} {:<15s} {:<20s} {:<5d} {:<10d} {:<10d} {:<10d} {:<10.2e} {:<15.2e} {:<15.2e} {:<15.2e} {:<15.2e} {:<10.2e}"
        .format(ProblemType(test_id).name, polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D(mesh_type).name,
                polydim.pde_tools.local_space_df_pcc_2_d.MethodTypes(method_id).name, method_order, mesh.cell2_d_total_number(),
                count_do_fs_data.num_total_do_fs, count_do_fs_data.num_total_strongs,
                post_process_data.mesh_size, post_process_data.error_l2_pressure, post_process_data.error_h1_velocity,
                post_process_data.norm_l2_pressure, post_process_data.norm_h1_velocity, post_process_data.residual_norm))

    file_name = file_path + "/Errors_" + str(test_id) + "_" + str(method_id) + "_" + str(method_order) + ".csv"
    if not os.path.exists(file_name):
        with open(file_name, 'w') as fd:
            fd.write('Test' + file_separator + 'Mesh' + file_separator + 'Method' + file_separator + 'Order' + file_separator + 'Cell2Ds' +
                     file_separator + 'DOFs' + file_separator + 'Strong' + file_separator + 'h' + file_separator
             + 'errorL2Pressure' + file_separator + 'errorH1Velocity' + file_separator + 'normL2Pressure' + file_separator + 'normH1Velocity' + file_separator + 'residual\n')

    with open(file_name, 'a') as fd:
        fd.write("{:<d}{:<s}{:<d}{:<s}{:<d}{:<s}{:<d}{:<s}{:<d}{:<s}{:<d}{:<s}{:<d}{:<s}{:<.16e}{:<s}{:<.16e}{:<s}{:<.16e}{:<s}{:<.16e}{:<s}{:<.16e}{:<s}{:<.16e}\n"
        .format(test_id, file_separator, mesh_type, file_separator, method_id, file_separator, method_order, file_separator,
                mesh.cell2_d_total_number(), file_separator,
                count_do_fs_data.num_total_do_fs, file_separator, count_do_fs_data.num_total_strongs, file_separator,
                post_process_data.mesh_size, file_separator, post_process_data.error_l2_pressure, file_separator, post_process_data.error_h1_velocity, file_separator,
                post_process_data.norm_l2_pressure, file_separator, post_process_data.norm_h1_velocity, file_separator, post_process_data.residual_norm))


    print('\x1b[35m' + "Errors exported in: " + file_name + '\x1b[0m')