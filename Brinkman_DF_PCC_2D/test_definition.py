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

from abc import ABC, abstractmethod
import numpy as np
from pypolydim import polydim
from typing import List, Dict
from enum import Enum

class ITest(ABC):

    @staticmethod
    @abstractmethod
    def domain() -> polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D:
        pass

    @staticmethod
    @abstractmethod
    def boundary_info() -> List[Dict[int, polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo]]:
        pass

    @abstractmethod
    def fluid_viscosity(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_diffusion_term(self, points: np.ndarray) -> List[np.ndarray]:
        pass

    @abstractmethod
    def source_term(self, points: np.ndarray) -> List[np.ndarray]:
        pass

    @abstractmethod
    def divergence_term(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def strong_boundary_condition(self, marker: int, points: np.ndarray) -> List[np.ndarray]:
        pass

    @abstractmethod
    def weak_boundary_condition(self, marker: int, points: np.ndarray) -> List[np.ndarray]:
        pass

    @abstractmethod
    def exact_pressure(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def exact_velocity(self, points: np.ndarray) -> List[np.ndarray]:
        pass

    @abstractmethod
    def exact_derivatives_velocity(self, points: np.ndarray) -> List[np.ndarray]:
        pass


class ProblemType(Enum):
    StokesSinSin = 1
    Brinkman = 2

class StokesSinSin(ITest):

    @staticmethod
    def domain() -> polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D:
        pde_domain = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D()

        pde_domain.vertices = np.array([[0.0, 1.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0]])

        pde_domain.area = 1.0
        pde_domain.shape_type = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D.Domain_Shape_Types.parallelogram

        return pde_domain

    @staticmethod
    def boundary_info() -> List[Dict[int, polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo]]:

        info_internal = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
        info_internal.marker = 0

        info_dirichlet = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(
            polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.strong)
        info_dirichlet.marker = 1

        info_neumann_bottom = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(
            polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.weak)
        info_neumann_bottom.marker = 2

        result = {
            0: info_internal,
            1: info_dirichlet,
            2: info_dirichlet,
            3: info_dirichlet,
            4: info_dirichlet,
            5: info_neumann_bottom,
            6: info_dirichlet,
            7: info_dirichlet,
            8: info_dirichlet
        }

        return [result, result]

    def fluid_viscosity(self, points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[1])

    def inverse_diffusion_term(self, points: np.ndarray) -> List[np.ndarray]:
        return [np.zeros(points.shape[1]), np.zeros(points.shape[1]), np.zeros(points.shape[1]),
                np.zeros(points.shape[1]), np.zeros(points.shape[1]), np.zeros(points.shape[1]),
                np.zeros(points.shape[1]), np.zeros(points.shape[1]), np.zeros(points.shape[1])]

    def source_term(self, points: np.ndarray) -> List[np.ndarray]:
        
        laplacian = [np.zeros(points.shape[1]), np.zeros(points.shape[1])]
        x = points[0]
        y = points[1]
        laplacian[0] = np.cos(y) * np.sin(y) * (-np.sin(x) * np.sin(x) + 3.0 * np.cos(x) * np.cos(x))
        laplacian[1] = -np.cos(x) * np.sin(x) * (-np.sin(y) * np.sin(y) + 3.0 * np.cos(y) * np.cos(y))

        pressure_derivatives = [np.zeros(points.shape[1]), np.zeros(points.shape[1])]
        pressure_derivatives[0] = np.cos(x)
        pressure_derivatives[1] = -np.cos(y)

        return [-laplacian[0] + pressure_derivatives[0],
                -laplacian[1] + pressure_derivatives[1],
                np.zeros(points.shape[1])]

    def divergence_term(self, points: np.ndarray) -> np.ndarray:
        return np.zeros(points.shape[1])

    def strong_boundary_condition(self, marker: int, points: np.ndarray) -> List[np.ndarray]:

        if marker != 1:
            raise ValueError("not valid marker")

        return self.exact_velocity(points)

    def weak_boundary_condition(self, marker: int, points: np.ndarray) -> List[np.ndarray]:
        x = points[0]
        y = points[1]

        match marker:
            case 2:
                return [-0.5 * np.cos(x) * np.cos(x) * (-np.sin(y) * np.sin(y) + np.cos(y) * np.cos(y)),
                        -np.cos(y) * np.sin(y) * np.cos(x) * np.sin(x) - np.sin(x) - np.sin(y),
                        np.zeros(points.shape[1])]
            case _:
                raise ValueError("unknown marker")

    def exact_pressure(self, points: np.ndarray) -> np.ndarray:
        x = points[0]
        y = points[1]

        return np.sin(x) - np.sin(y)

    def exact_velocity(self, points: np.ndarray) -> List[np.ndarray]:
        x = points[0]
        y = points[1]

        return [-0.5 * np.cos(x) * np.cos(x) * np.cos(y) * np.sin(y),
                0.5 * np.cos(y) * np.cos(y) * np.cos(x) * np.sin(x),
                np.zeros(points.shape[1])]

    def exact_derivatives_velocity(self, points: np.ndarray) -> List[np.ndarray]:
        x = points[0]
        y = points[1]

        return [np.cos(x) * np.sin(x) * np.cos(y) * np.sin(y),
                -0.5 * np.cos(x) * np.cos(x) * (-np.sin(y) * np.sin(y) + np.cos(y) * np.cos(y)),
                np.zeros(points.shape[1]),
                0.5 * np.cos(y) * np.cos(y) * (-np.sin(x) * np.sin(x) + np.cos(x) * np.cos(x)),
                -np.cos(y) * np.sin(y) * np.cos(x) * np.sin(x),
                np.zeros(points.shape[1]),
                np.zeros(points.shape[1]),
                np.zeros(points.shape[1]),
                np.zeros(points.shape[1])]

class Brinkman(ITest):

    @staticmethod
    def domain() -> polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D:
        pde_domain = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D()

        pde_domain.vertices = np.array([[0.0, 1.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0]])

        pde_domain.area = 1.0
        pde_domain.shape_type = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D.Domain_Shape_Types.parallelogram

        return pde_domain

    @staticmethod
    def boundary_info() -> List[Dict[int, polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo]]:

        info_internal = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
        info_internal.marker = 0

        info_dirichlet = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(
            polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.strong)
        info_dirichlet.marker = 1

        result = {
            0: info_internal,
            1: info_dirichlet,
            2: info_dirichlet,
            3: info_dirichlet,
            4: info_dirichlet,
            5: info_dirichlet,
            6: info_dirichlet,
            7: info_dirichlet,
            8: info_dirichlet
        }

        return [result, result]

    def fluid_viscosity(self, points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[1])

    def inverse_diffusion_term(self, points: np.ndarray) -> List[np.ndarray]:
        return [np.ones(points.shape[1]), np.zeros(points.shape[1]), np.zeros(points.shape[1]),
                np.zeros(points.shape[1]), np.ones(points.shape[1]), np.zeros(points.shape[1]),
                np.zeros(points.shape[1]), np.zeros(points.shape[1]), np.zeros(points.shape[1])]

    def source_term(self, points: np.ndarray) -> List[np.ndarray]:
        x = points[0]
        y = points[1]
        return [(2.0 * np.pi * np.pi + 1.0) * np.sin(np.pi * x) * np.cos(np.pi * y ) + 2.0 * x * y  * y,
                (-2.0 * np.pi * np.pi - 1.0) * np.cos(np.pi * x) * np.sin(np.pi * y) + 2.0 * y  * x * x,
                np.zeros(points.shape[1])]

    def divergence_term(self, points: np.ndarray) -> np.ndarray:
        return np.zeros(points.shape[1])
    

    def strong_boundary_condition(self, marker: int, points: np.ndarray) -> List[np.ndarray]:
        if marker != 1:
            raise ValueError("Unknown marker")

        x = points[0]
        y = points[1]

        return [np.sin(np.pi * x) * np.cos(np.pi * y),
                -np.cos(np.pi * x) * np.sin(np.pi * y),
                np.zeros(points.shape[1])]


    def weak_boundary_condition(self, marker: int, points: np.ndarray) -> List[np.ndarray]:
        match marker:
            case _:
                raise ValueError("unknown marker")


    def exact_pressure(self, points: np.ndarray) -> np.ndarray:
        
        x = points[0]
        y = points[1]
        
        return x * x * y  * y  - 1.0 / 9.0

    def exact_velocity(self, points: np.ndarray) -> List[np.ndarray]:
        x = points[0]
        y = points[1]
        return [np.sin(np.pi * x) * np.cos(np.pi * y ),
                -np.cos(np.pi * x) * np.sin(np.pi * y ),
                np.zeros(points.shape[1])]
    

    def exact_derivatives_velocity(self, points: np.ndarray) -> List[np.ndarray]:

        x = points[0]
        y = points[1]

        return [np.pi * np.cos(np.pi * x) * np.cos(np.pi * y),
                -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y),
                np.zeros(points.shape[1]),
                np.pi * np.sin(np.pi * x) * np.sin(np.pi * y),
                -np.pi * np.cos(np.pi * x) * np.cos(np.pi * y),
                np.zeros(points.shape[1]),
                np.zeros(points.shape[1]),
                np.zeros(points.shape[1]),
                np.zeros(points.shape[1])]
