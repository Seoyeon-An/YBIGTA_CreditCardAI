import math
import numpy as np
import torch
import torch.nn.functional as F

class TS_SS:
    def __init__(self, device: torch.device):
        self.device = device

    def Cosine(self, vec1: torch.Tensor, vec2: torch.Tensor):
        m = torch.matmul(vec1, vec2.T)/(torch.linalg.norm(vec1))
        dig = torch.diagonal(m, 0)
        norm = torch.linalg.norm(vec2)
        # print(f"DIG: {dig}")
        # print(f"NORM: {norm}")
        result = dig * norm
        return result

    def VectorSize(self, vec: torch.Tensor):
        return torch.linalg.norm(vec, dim=-1) # (10, 1)?

    def Euclidean(self, vec1: torch.Tensor, vec2: torch.Tensor):
        return torch.linalg.norm(vec1-vec2)

    def Theta(self, vec1: torch.Tensor, vec2: torch.Tensor):
        arc = torch.acos(F.cosine_similarity(vec1, vec2))
        # print(f"ARC: {arc}")
        deg = torch.deg2rad(torch.tensor([10]).to(self.device))
        return arc + deg

    def Triangle(self, vec1: torch.Tensor, vec2: torch.Tensor):
        # intput: (10, 30)
        # output: (10, 1)
        t = self.Theta(vec1, vec2)
        # print(f"theta: {t}")
        theta = torch.deg2rad(self.Theta(vec1, vec2))
        # print(f"Theta: {theta.size()}")
        # vec1_size = self.VectorSize(vec1)
        # print(f'vec1_size: {vec1_size.size()}')
        # result = (self.VectorSize(vec1) * self.VectorSize(vec2) * torch.sin(theta))/2
        # print(f"Result size: {result.size()}")
        return (self.VectorSize(vec1) * self.VectorSize(vec2) * torch.sin(theta))/2

    def Magnitude_Difference(self, vec1: torch.Tensor, vec2: torch.Tensor):
        return abs(self.VectorSize(vec1) - self.VectorSize(vec2))

    def Sector(self, vec1: torch.Tensor, vec2: torch.Tensor):
        ED = self.Euclidean(vec1, vec2)
        MD = self.Magnitude_Difference(vec1, vec2)
        theta = self.Theta(vec1, vec2)
        return math.pi * (ED + MD)**2 * theta/360


    def __call__(self, vec1: torch.Tensor, vec2: torch.Tensor):
        # tri = self.Triangle(vec1, vec2)
        # sec = self.Sector(vec1, vec2)
        # print(f"Triangle: {tri}")
        # print(f"Sector: {sec}")

        # print(f"Tri size: {tri.size()}")
        # print(f"Sec size: {sec.size()}")
        return self.Triangle(vec1, vec2) * self.Sector(vec1, vec2)

if __name__ == '__main__':
    # Usage
    v1 = torch.rand(10, 30)
    v2 = torch.rand(10, 30)
    similarity = TS_SS('cpu')
    sim = similarity(v1,v2)
    print(f"Sim size: {sim.size()}")

    # to convert to tensor
    