import torch
from typing import Optional, Tuple


class QuaternionOperations:
    """Quaternion operators with batch-aware numerical safeguards.

    The class exposes quaternion algebra (Hamilton product, log/exp maps, averaging),
    hemisphere alignment utilities, and SO(3) Jacobians required by Kalman filtering.
    All routines accept tensors with arbitrary leading batch dimensions and expect the
    quaternion component axis to have length four arranged as ``(w, x, y, z)``.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        """Create a utility instance.

        Args:
            epsilon: Base tolerance used whenever small denominators appear.
        """
        self.epsilon = epsilon

    def _eps(self, tensor: torch.Tensor) -> float:
        """Return a dtype-aware epsilon for numerical safeguards.

        Args:
            tensor: Tensor providing the floating-point dtype context. Shape is arbitrary.

        Returns:
            Scalar tolerance compatible with ``tensor``'s dtype.
        """
        if tensor.is_floating_point():
            return max(self.epsilon, torch.finfo(tensor.dtype).eps)
        return self.epsilon

    def _canonicalize_dim(self, dim: int, ndim: int) -> int:
        """Normalize possibly-negative dimension indices and validate bounds.

        Args:
            dim: Possibly negative axis index.
            ndim: Total number of axes in the tensor the index refers to.

        Returns:
            Non-negative axis index within ``[0, ndim)``.

        Raises:
            ValueError: If ``dim`` is out of range after normalization.
        """
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"Dimension index {dim} is out of bounds for tensor with {ndim} dims")
        return dim

    def _prepare_weights(
        self,
        weights: Optional[torch.Tensor],
        canonical_dim: int,
        quaternion_ndim: int,
        target_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Broadcast and validate weight tensors to match quaternion batches.

        Args:
            weights: Optional tensor whose trailing averaging dimension has length ``target_length``.
                Supported layouts are ``(..., target_length)`` or ``(..., target_length, 1)`` where
                the ellipsis matches the quaternion batch shape after moving ``canonical_dim`` to
                the second-to-last axis.
            canonical_dim: Dimension index in the original quaternion tensor that contains the
                elements to be averaged.
            quaternion_ndim: Rank of the quaternion tensor prior to any dimension moves.
            target_length: Size of the averaging axis after canonicalization.
            device: Target device for the returned tensor.
            dtype: Target dtype for the returned tensor.

        Returns:
            Tensor broadcast to shape ``(..., target_length, 1)`` on the requested device/dtype, or
            ``None`` if ``weights`` is ``None``.
        """
        if weights is None:
            return None

        if weights.dim() not in {quaternion_ndim, quaternion_ndim - 1}:
            raise ValueError(
                "Weights must have either the same rank as quaternions or one fewer (missing the quaternion component)"
            )

        weight_dim = self._canonicalize_dim(canonical_dim, weights.dim())

        if weights.dim() == quaternion_ndim:
            weights = torch.movedim(weights, weight_dim, weights.dim() - 2)
            if weights.shape[-1] != 1:
                raise ValueError("Weights with the same rank as quaternions must have a trailing dimension of size 1")
        else:
            weights = torch.movedim(weights, weight_dim, weights.dim() - 1)
            weights = weights.unsqueeze(-1)

        if weights.shape[-2] != target_length:
            raise ValueError(
                f"Weights length {weights.shape[-2]} does not match quaternion batch length {target_length}"
            )

        return weights.to(device=device, dtype=dtype)

    def _prepare_mean_inputs(
        self,
        quaternions: torch.Tensor,
        weights: Optional[torch.Tensor],
        dim: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Normalize, align, and broadcast quaternions (and weights) for averaging.

        Args:
            quaternions: Tensor with shape ``(..., N, 4)`` once ``dim`` is moved to the second-to-last
                axis. The last dimension always stores ``(w, x, y, z)`` components.
            weights: Optional tensor aligned with the averaging axis ``N``. See ``_prepare_weights``
                for accepted layouts.
            dim: Dimension index of ``quaternions`` that enumerates the samples to be averaged.

        Returns:
            Tuple ``(quats, weights)`` where ``quats`` is normalized and hemisphere-consistent with
            shape ``(..., N, 4)`` and ``weights`` is either ``None`` or has shape ``(..., N, 1)``.
        """
        if quaternions.numel() == 0:
            raise ValueError("Input quaternion tensor cannot be empty")

        self.validate_quaternion(quaternions)

        canonical_dim = self._canonicalize_dim(dim, quaternions.dim())
        if canonical_dim == quaternions.dim() - 1:
            raise ValueError("Averaging dimension cannot be the last quaternion component axis")

        quats = torch.movedim(quaternions, canonical_dim, quaternions.dim() - 2)
        quats = self.normalize(quats)
        quats = self.ensure_hemisphere_consistency(quats)

        broadcast_weights = self._prepare_weights(
            weights,
            canonical_dim,
            quaternions.dim(),
            quats.shape[-2],
            quats.device,
            quats.dtype,
        )

        return quats, broadcast_weights

    def validate_quaternion(self, quaternion: torch.Tensor) -> None:
        """
        Validate that the input tensor has the correct shape for quaternion representation.

        Args:
            quaternion: Tensor whose last axis is expected to encode ``(w, x, y, z)``.

        Raises:
            ValueError: If ``quaternion.shape[-1]`` is not equal to four.
        """
        if quaternion.shape[-1] != 4:
            raise ValueError(
                f"Quaternion tensor must have last dimension of size 4, got {quaternion.shape}"
            )

    def conjugate(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Compute the conjugate of a quaternion.

        For a quaternion q = (w, x, y, z), the conjugate is q* = (w, -x, -y, -z).
        For unit quaternions, the conjugate is equal to the inverse.

        Args:
            quaternion: Tensor ``(..., 4)`` containing quaternions ordered as ``(w, x, y, z)``.

        Returns:
            Tensor ``(..., 4)`` with conjugated quaternions.
        """
        self.validate_quaternion(quaternion)
        return torch.stack(
            (quaternion[..., 0], -quaternion[..., 1], -quaternion[..., 2], -quaternion[..., 3]),
            dim=-1
        )

    def hamilton_product(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamilton product of two quaternions with broadcasting support.

        The Hamilton product is defined as:
        q1 ⊗ q2 = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
        where q1 = (w1, v1) and q2 = (w2, v2)

        Args:
            q1: Tensor ``(..., 4)`` storing the left factors in ``(w, x, y, z)`` order.
            q2: Tensor ``(..., 4)`` storing the right factors. Broadcasting follows PyTorch rules.

        Returns:
            Tensor ``(..., 4)`` containing ``q1 ⊗ q2`` after broadcasting.
        """
        self.validate_quaternion(q1)
        self.validate_quaternion(q2)

        # Broadcast tensors to compatible shapes
        q1, q2 = torch.broadcast_tensors(q1, q2)

        # Extract components
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)

        # Compute Hamilton product
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack((w, x, y, z), dim=-1)

    def normalize(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Normalize quaternions to unit length with numerical stability.

        Args:
            quaternion: Tensor ``(..., 4)`` representing arbitrary quaternions.

        Returns:
            Tensor ``(..., 4)`` where each quaternion has unit Euclidean norm.
        """
        self.validate_quaternion(quaternion)
        norm = quaternion.norm(dim=-1, keepdim=True).clamp_min(self._eps(quaternion))
        return quaternion / norm

    def quaternion_logarithm(self, quaternion: torch.Tensor, handle_pi: bool) -> torch.Tensor:
        """Quaternion logarithm with optional stability near ``π`` rotations.

        Args:
            quaternion: Tensor ``(..., 4)`` containing unit (or normalizable) quaternions.
            handle_pi: When ``True`` forces a stable axis selection for 180° rotations.

        Returns:
            Tensor ``(..., 4)`` whose scalar component is zero and whose vector component encodes
            half-angle rotation vectors ``φ = θ/2 · u``.
        """
        q_normalized = self.normalize(quaternion)

        w = q_normalized[..., 0].clamp(-1.0, 1.0)
        xyz = q_normalized[..., 1:]
        xyz_norm = xyz.norm(dim=-1, keepdim=True)

        angle = torch.atan2(xyz_norm, w.unsqueeze(-1))
        eps = self._eps(q_normalized)
        axis = xyz / xyz_norm.clamp_min(eps)
        log_xyz = axis * angle

        small_rotation_mask = (xyz_norm < eps).expand_as(log_xyz)
        log_xyz = torch.where(small_rotation_mask, xyz, log_xyz)

        if handle_pi:
            near_pi_mask = (w.abs() < eps) & (xyz_norm.squeeze(-1) > 1.0 - eps)
            if near_pi_mask.any():
                abs_xyz = xyz.abs()
                max_component_idx = abs_xyz.argmax(dim=-1, keepdim=True)
                gathered = xyz.gather(-1, max_component_idx)
                axis_sign = torch.where(gathered >= 0.0, torch.ones_like(gathered), -torch.ones_like(gathered))
                # Flip the entire axis based on the dominant component's sign for stability
                stabilized_axis = axis * axis_sign
                stabilized_log = stabilized_axis * angle
                log_xyz = torch.where(near_pi_mask.unsqueeze(-1), stabilized_log, log_xyz)

        zeros = torch.zeros_like(angle)
        return torch.cat((zeros, log_xyz), dim=-1)

    def quaternion_exponential(self, pure_quaternion: torch.Tensor) -> torch.Tensor:
        """
        Apply exponential map to pure quaternions, returning unit quaternions.

        The exponential map converts a rotation vector to a unit quaternion.
        For v_q = (0, v), exp(v_q) = (cos(||v||), v/||v|| * sin(||v||)).

        Args:
            pure_quaternion: Tensor ``(..., 4)`` whose scalar component is zero and vector component
                stores half-angle rotation vectors ``φ``.

        Returns:
            Tensor ``(..., 4)`` representing the unit quaternions ``exp(pure_quaternion)``.
        """
        self.validate_quaternion(pure_quaternion)

        # Extract vector part
        v = pure_quaternion[..., 1:]
        angle = v.norm(dim=-1, keepdim=True)
        angle_sq = angle.pow(2)

        # Compute sine and cosine of the angle
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)

        # Handle small angle case using Taylor series approximation
        eps = self._eps(angle)
        small_angle_mask = angle < eps

        # For small angles: sin(angle)/angle ≈ 1 - angle²/6
        sin_scale = torch.where(
            small_angle_mask,
            1.0 - angle_sq / 6.0,
            sin_angle / angle.clamp_min(eps),
        )

        # For small angles: cos(angle) ≈ 1 - angle²/2
        w = torch.where(
            small_angle_mask,
            1.0 - angle_sq / 2.0,
            cos_angle,
        )

        # Compute vector part
        xyz = v * sin_scale

        # Combine into quaternion
        quaternion = torch.cat((w, xyz), dim=-1)
        return self.normalize(quaternion)

    def ensure_hemisphere_consistency(self, quaternion_sequence: torch.Tensor) -> torch.Tensor:
        """
        Ensure quaternion sequence stays in the same hemisphere to avoid sign flips.

        Since q and -q represent the same rotation, this function ensures continuity
        in the quaternion sequence by flipping signs when necessary.

        Args:
            quaternion_sequence: Tensor ``(..., T, 4)`` listing quaternions over time (or any
                ordered axis) with ``(w, x, y, z)`` components.

        Returns:
            Tensor ``(..., T, 4)`` containing hemisphere-consistent unit quaternions.
        """
        self.validate_quaternion(quaternion_sequence)

        # Normalize all quaternions and compute sign corrections in a vectorized fashion
        normalized_sequence = self.normalize(quaternion_sequence)

        if normalized_sequence.shape[-2] <= 1:
            return normalized_sequence

        dot_products = (normalized_sequence[..., 1:, :] * normalized_sequence[..., :-1, :]).sum(dim=-1)
        flip_indicator = torch.where(dot_products < 0.0, -1.0, 1.0)

        leading = torch.ones(*normalized_sequence.shape[:-2], 1, device=normalized_sequence.device, dtype=normalized_sequence.dtype)
        signs = torch.cumprod(torch.cat((leading, flip_indicator), dim=-1), dim=-1)

        return normalized_sequence * signs.unsqueeze(-1)

    def align_with_reference(self, quaternion: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Align quaternion with a reference to ensure they are in the same hemisphere.

        Args:
            quaternion: Tensor ``(..., 4)`` containing quaternions to align.
            reference: Tensor broadcastable to ``quaternion`` specifying the desired hemisphere.

        Returns:
            Tensor ``(..., 4)`` equal to ``quaternion`` or its negation, aligned with ``reference``.
        """
        self.validate_quaternion(quaternion)
        self.validate_quaternion(reference)

        quaternion, reference = torch.broadcast_tensors(quaternion, reference)
        dot_product = (quaternion * reference).sum(dim=-1, keepdim=True)
        sign = torch.where(dot_product < 0.0, -1.0, 1.0)
        return quaternion * sign

    def to_pure_quaternion(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Convert a quaternion to a pure quaternion by zeroing out the scalar part.

        Args:
            quaternion: Tensor ``(..., 4)`` with ``(w, x, y, z)`` ordering.

        Returns:
            Tensor ``(..., 4)`` whose scalar component is zero and vector part is unchanged.
        """
        self.validate_quaternion(quaternion)
        zeros = torch.zeros_like(quaternion[..., :1])
        return torch.cat((zeros, quaternion[..., 1:]), dim=-1)


    # ----------------------------
    # Quaternion Averaging Methods
    # ----------------------------
    def _compute_weighted_average(
        self,
        tensors: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: int = -2
    ) -> torch.Tensor:
        """
        Compute weighted average along specified dimension.

        This is a generic weighted averaging method that can be applied to any tensor.
        It handles weight normalization, broadcasting, and dimension management.

        Args:
            tensors: Tensor ``(..., N, D)`` after moving the averaging axis to position ``dim``.
            weights: Optional tensor ``(..., N, 1)`` providing per-sample weights. The final
                dimension ``1`` is broadcast across ``D``.
            dim: Dimension index holding the ``N`` samples. Only ``-2`` is used internally.

        Returns:
            Tensor ``(..., D)`` containing the weighted averages.
        """
        if weights is None:
            # Perform uniform averaging
            return torch.mean(tensors, dim=dim)

        # Normalize weights along the averaging dimension
        weight_sum = torch.sum(weights, dim=dim, keepdim=True)
        weight_sum = weight_sum.clamp_min(self._eps(weights))
        normalized_weights = weights / weight_sum

        # Perform weighted averaging
        return torch.sum(tensors * normalized_weights, dim=dim)

    def quaternion_to_log_space_vector(
        self,
        quaternions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map quaternions to log space (tangent space) as 3D vectors.

        This method extracts the rotation vector from a unit quaternion by applying
        the logarithm map and returning only the vector part.

        Args:
            quaternions: Tensor ``(..., 4)`` with unit (or normalizable) quaternions.

        Returns:
            Tensor ``(..., 3)`` containing half-angle rotation vectors ``φ``.
        """
        # Apply logarithm map to get pure quaternion
        log_quaternions = self.quaternion_logarithm(quaternions, handle_pi=True)

        # Extract vector part (rotation vector)
        return log_quaternions[..., 1:]

    def log_space_vector_to_quaternion(self, log_vectors: torch.Tensor) -> torch.Tensor:
        """
        Map log space vectors back to quaternion space.

        This method converts a 3D rotation vector to a pure quaternion and then
        applies the exponential map to get a unit quaternion.

        Args:
            log_vectors: Tensor ``(..., 3)`` with half-angle rotation vectors ``φ``.

        Returns:
            Tensor ``(..., 4)`` of unit quaternions reconstructed via the exponential map.
        """
        # Create pure quaternion by adding zero scalar part
        zeros = torch.zeros_like(log_vectors[..., :1])
        pure_quaternions = torch.cat((zeros, log_vectors), dim=-1)

        # Apply exponential map to get unit quaternion
        return self.quaternion_exponential(pure_quaternions)

    def arithmetic_mean(
        self,
        quaternions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: int = -2
    ) -> torch.Tensor:
        """
        Arithmetic averaging method: Fast and suitable for most applications.

        This method computes quaternion average through weighted arithmetic averaging
        followed by normalization. For rotations with small angular differences,
        the result is very close to the geometric mean.

        Advantages:
        - Fastest computation speed
        - Simple implementation
        - Numerically stable
        - Supports weighted averaging

        Disadvantages:
        - Less rigorous geometric meaning for large angular differences

        Args:
            quaternions: Tensor whose ``dim``-th axis enumerates ``N`` quaternions; after internal
                reordering the data has shape ``(..., N, 4)``.
            weights: Optional tensor broadcastable to ``(..., N, 1)`` providing averaging weights.
            dim: Axis that lists the samples being averaged (must not be the component axis).

        Returns:
            Tensor ``(..., 4)`` containing normalized arithmetic means.

        Raises:
            ValueError: If ``quaternions`` is empty or the component axis is mis-specified.
        """
        processed_quaternions, processed_weights = self._prepare_mean_inputs(
            quaternions, weights, dim
        )

        # Perform weighted averaging
        mean_quaternion = self._compute_weighted_average(processed_quaternions, processed_weights, -2)

        # Normalize result
        return self.normalize(mean_quaternion)

    def log_space_mean(
        self,
        quaternions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: int = -2
    ) -> torch.Tensor:
        """
        Log-space averaging method: Geometrically rigorous and precise.

        This method maps quaternions to log space (tangent space), performs weighted
        arithmetic averaging in that space, then maps back to quaternion space.
        This provides true geometric averaging.

        Advantages:
        - Geometrically rigorous
        - Order-independent
        - Suitable for arbitrary angular differences
        - Supports weighted averaging

        Disadvantages:
        - Slightly higher computational complexity

        Args:
            quaternions: Tensor whose ``dim``-th axis enumerates ``N`` quaternions; internally moved
                to ``(..., N, 4)``.
            weights: Optional tensor broadcastable to ``(..., N, 1)`` providing averaging weights.
            dim: Axis that lists the samples being averaged (must not be the component axis).

        Returns:
            Tensor ``(..., 4)`` containing log-space (geometric) means.

        Raises:
            ValueError: If ``quaternions`` is empty or the component axis is mis-specified.
        """
        processed_quaternions, processed_weights = self._prepare_mean_inputs(
            quaternions, weights, dim
        )

        # Map to log space as 3D vectors
        log_vectors = self.quaternion_to_log_space_vector(processed_quaternions)

        # Compute weighted mean in log space
        mean_log = self._compute_weighted_average(log_vectors, processed_weights, -2)

        # Map back to quaternion space
        mean_quaternion = self.log_space_vector_to_quaternion(mean_log)

        # Normalize result
        return self.normalize(mean_quaternion)

    def compute_mean(
        self,
        quaternions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        method: str = 'arithmetic',
        dim: int = -2
    ) -> torch.Tensor:
        """
        Generic quaternion averaging interface with weight support.

        Args:
            quaternions: Tensor whose ``dim``-th axis enumerates ``N`` quaternions.
            weights: Optional tensor broadcastable to ``(..., N, 1)`` containing sample weights.
            method: Either ``'arithmetic'`` or ``'log_space'``.
            dim: Axis that lists the samples being averaged.

        Returns:
            Tensor ``(..., 4)`` containing the selected mean.

        Raises:
            ValueError: If ``method`` is not supported.
        """
        if method == 'arithmetic':
            return self.arithmetic_mean(quaternions, weights, dim)
        elif method == 'log_space':
            return self.log_space_mean(quaternions, weights, dim)
        else:
            raise ValueError(f"Unsupported averaging method: {method}. Supported methods: 'arithmetic', 'log_space'")

    # ----------------------------
    # Lie Group Operations for Kalman Filtering
    # ----------------------------
    def _skew_symmetric(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Create a skew-symmetric matrix from a 3D vector.

        For a vector v = [x, y, z], the skew-symmetric matrix is:
        [ 0  -z   y]
        [ z   0  -x]
        [-y   x   0]

        Args:
            vector: Tensor ``(..., 3)`` containing 3D vectors.

        Returns:
            Tensor ``(..., 3, 3)`` representing skew-symmetric matrices.
        """
        x, y, z = vector.unbind(dim=-1)
        zeros = torch.zeros_like(x)

        skew = torch.stack([
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1)
        ], dim=-2)

        return skew

    def left_jacobian(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the left Jacobian J_l(phi) for SO(3) Lie group.

        The left Jacobian maps from tangent space to the Lie group and is used
        for proper linearization in the extended Kalman filter on manifolds.

        Args:
            phi: Tensor ``(..., 3)`` with rotation vectors in the tangent space.

        Returns:
            Tensor ``(..., 3, 3)`` containing the corresponding left Jacobian matrices.
        """
        angle = phi.norm(dim=-1, keepdim=True)
        eps = self._eps(phi)

        I3 = torch.eye(3, device=phi.device, dtype=phi.dtype)
        Phi_x = self._skew_symmetric(phi)
        Phi_x2 = torch.matmul(Phi_x, Phi_x)

        small_angle_mask = (angle < eps).squeeze(-1)
        result_small = I3 + 0.5 * Phi_x + (1.0 / 6.0) * Phi_x2

        theta = angle.clamp_min(eps)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        inv_theta_sq = (1.0 / (theta * theta)).unsqueeze(-1)
        inv_theta_cu = (1.0 / (theta.pow(3))).unsqueeze(-1)

        A = (1.0 - cos_theta).unsqueeze(-1) * inv_theta_sq
        B = (theta - sin_theta).unsqueeze(-1) * inv_theta_cu
        result_large = I3 + A * Phi_x + B * Phi_x2

        mask = small_angle_mask.unsqueeze(-1).unsqueeze(-1)
        return torch.where(mask, result_small, result_large)

    def left_jacobian_inverse(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of the left Jacobian J_l⁻¹(phi) for SO(3) Lie group.

        Args:
            phi: Tensor ``(..., 3)`` with rotation vectors in the tangent space.

        Returns:
            Tensor ``(..., 3, 3)`` containing the inverse Jacobian matrices.
        """
        angle = phi.norm(dim=-1, keepdim=True)
        eps = self._eps(phi)

        I3 = torch.eye(3, device=phi.device, dtype=phi.dtype)
        Phi_x = self._skew_symmetric(phi)
        Phi_x2 = torch.matmul(Phi_x, Phi_x)
        small_angle_mask = (angle < eps).squeeze(-1)
        result_small = I3 - 0.5 * Phi_x + (1.0 / 12.0) * Phi_x2

        theta = angle.clamp_min(eps)
        half_theta = 0.5 * theta
        sin_half = torch.sin(half_theta).clamp_min(eps)
        cot_half = torch.cos(half_theta) / sin_half
        a = half_theta * cot_half

        phi_outer = phi.unsqueeze(-1) @ phi.unsqueeze(-2)
        coeff = ((1.0 - a) / (theta * theta)).unsqueeze(-1)
        a_expanded = a.unsqueeze(-1)

        result_large = a_expanded * I3 - 0.5 * Phi_x + coeff * phi_outer

        mask = small_angle_mask.unsqueeze(-1).unsqueeze(-1)
        return torch.where(mask, result_small, result_large)

    def quaternion_to_rotation_matrix(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Convert a unit quaternion to a rotation matrix.

        Args:
            quaternion: Tensor ``(..., 4)`` containing unit (or normalizable) quaternions ordered
                as ``(w, x, y, z)``.

        Returns:
            Tensor ``(..., 3, 3)`` with rotation matrices.
        """
        q = self.normalize(quaternion)
        w, x, y, z = q.unbind(dim=-1)

        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        # Construct rotation matrix
        R = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
        ], dim=-2)

        return R
