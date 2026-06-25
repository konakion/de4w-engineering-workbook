import requests


class PhyphoxClient:
    """
    Simple client for reading live accelerometer values from phyphox.

    phyphox must run on the smartphone with remote access enabled.
    The smartphone and laptop must be connected to the same network.
    """

    def __init__(self, base_url: str) -> None:
        """
        Parameters
        ----------
        base_url:
            URL shown by phyphox remote access, for example:
            http://192.168.178.45:8080
        """
        self.base_url = base_url.rstrip("/")

    def get_acceleration(self) -> tuple[float, float, float]:
        """
        Read the latest x, y, z acceleration values from phyphox.

        Returns
        -------
        tuple[float, float, float]
            Latest acceleration values for x, y, z.
        """
        url = f"{self.base_url}/get?accX&accY&accZ"

        response = requests.get(url, timeout=2)
        response.raise_for_status()

        data = response.json()

        x = data["buffer"]["accX"]["buffer"][-1]
        y = data["buffer"]["accY"]["buffer"][-1]
        z = data["buffer"]["accZ"]["buffer"][-1]

        return float(x), float(y), float(z)