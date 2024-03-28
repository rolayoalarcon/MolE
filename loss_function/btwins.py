import torch

class BarlowTwinsObjective(torch.nn.Module):

    """
    Computes the Barlow Twins loss function.

    Args:
        device (torch.device): The device to perform computations on.
        batch_size (int): Batch size used for training.
        l (float): Scaling parameter for the off-diagonal loss component.

    Attributes:
        batch_size (int): Batch size used for training.
        lambda_param (float): Scaling parameter for the off-diagonal loss component.
        device (torch.device): The device to perform computations on.

    """

    def __init__(self, device, batch_size, l):
        super(BarlowTwinsObjective, self).__init__()

        self.batch_size = batch_size
        self.lambda_param = l
        self.device = device
    
    def standardize(self, x):

        """
        Standardizes the input data.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Standardized tensor.
        """

        # First, we center by the mean
        #x_centered = x.sub(x.mean(axis=1).view(x.shape[0], 1))
        x_centered = x - x.mean(0)

        # Then, we divide by the standard deviation
        #x_standard = x_centered.div(x.std(axis=1).view(x.shape[0], 1))
        x_standard = x_centered / x.std(0)

        # Finished!
        return x_standard
    
    def correlation_matrix(self, x_base, x_noise):

        """
        Computes the correlation matrix between two sets of data.

        Args:
            x_base (Tensor): Base data.
            x_noise (Tensor): Noisy data.

        Returns:
            Tensor: Correlation matrix.
        """

        # Standardize
        x_base_standard = self.standardize(x_base)
        x_noise_standard = self.standardize(x_noise)

        # Matrix multiplication
        C = torch.mm(x_base_standard.T, x_noise_standard)

        # Normalizing range
        C_norm = C.div(self.batch_size)

        return C_norm

    def off_diagonal(self, x):

        """
        Returns the off-diagonal elements of a square matrix.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Flattened view of the off-diagonal elements.
        """

        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def lc_components(self, C):

        """
        Computes the on-diagonal and off-diagonal components of the loss.

        Args:
            C (Tensor): Correlation matrix.

        Returns:
            Tensor: On-diagonal components.
            Tensor: Off-diagonal components.
        """

        # Estimate diagonal distance to 1
        on_diagonal = torch.diagonal(C).add(-1).pow(2).sum()

        # Gather off-diagonal values
        off_diag = self.off_diagonal(x=C).pow(2).sum()

        return on_diagonal, off_diag

    def lc_loss(self, on_diagonal, off_diag):

        """
        Computes the final loss value.

        Args:
            on_diagonal (Tensor): On-diagonal components.
            off_diag (Tensor): Off-diagonal components.

        Returns:
            Tensor: Final loss value.
        """

        # Now we weight our pieces
        final_lc = on_diagonal + (self.lambda_param * off_diag)

        return final_lc

    def forward(self, z1, z2):

        """
        Forward pass of the Barlow Twins loss function.

        Args:
            z1 (Tensor): Representation from the first view.
            z2 (Tensor): Representation from the second view.

        Returns:
            Tensor: Loss value.
        """

        z1 = z1.float()
        z2 = z2.float()

        # Barlow Twins 
        corr_mat = self.correlation_matrix(x_base=z1, x_noise=z2)
        diagonal_elements, offDiag_elements = self.lc_components(C=corr_mat)
        loss = self.lc_loss(on_diagonal=diagonal_elements, off_diag=offDiag_elements)

        #print(f"D:{diagonal_elements} | OD:{offDiag_elements} | OD*lambda: {self.lambda_param * offDiag_elements}")

        return loss