
class Rescaler:
    def check_vals(
        self,
        max:float,
        xs:np.ndarray,
        check_max:bool,
    ):
        """Adjust the x values s.t. they are on a range from 0 to 1, where if x_i
        equals max then set that value to one

        Args:
            max (float): The maximum possible value
            xs (np.ndarray): The list of values
            check_max (bool): If True -> throw error if any values in xs are more
            than max, do no such error handling otherwise
        """
        assert np.all(xs >= 0), f"Scores must be nonnegative"
        if check_max and (np.max(xs) > (max + THIRTY_TWO_BIT_PRECISION)):
            raise ValueError(f"np.max(xs) = {np.max(xs)} > {max}")


    def rel_value(
        self,
            max:float,
            xs:np.ndarray,
            check_max:bool,
    ) -> np.ndarray:
        """Adjust the x values s.t. they are on a range from 0 to 1, where if x_i
        equals max then set that value to one

        Args:
            max (float): The maximum possible value
            xs (np.ndarray): The list of values
            check_max (bool): If True -> throw error if any values in xs are more
            than max, do no such error handling otherwise

        Returns:
            np.ndarray: Those adjust values
        """
        self.check_vals(
            max,
            xs,
            check_max,
        )

        return xs / max

    def rel_residue(
        self,
        max: float,
        xs:np.ndarray,
        check_max:bool,
    ) -> np.ndarray:
        """Calculate relative residual

        Args:
            max (float): The maximum possible value in xs

        Returns:
            residue (np.ndarray): The relative residue
            """
        self.check_vals(
            max,
            xs,
            check_max,
        )
        return np.absolute(max - xs) / max
