from typing import Any

import requests


PHYPHOX_NETWORK_HINT = (
    "Check that phyphox remote access is running and that the phone and laptop "
    "are on the same network. University WiFi such as eduroam may block "
    "device-to-device traffic; a phone hotspot or local router is often more reliable."
)


class PhyphoxClientError(RuntimeError):
    """
    Raised when live acceleration cannot be read from phyphox.
    """


class PhyphoxClient:
    """
    Client for reading live accelerometer values from phyphox.

    phyphox must run on the smartphone with remote access enabled.
    The smartphone and laptop must be connected to the same network.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 2.0) -> None:
        """
        Parameters
        ----------
        base_url:
            URL shown by phyphox remote access, for example:
            http://192.168.178.45:8080

        timeout_seconds:
            HTTP timeout for one phyphox request.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _read_latest_buffer_value(
        self,
        data: dict[str, Any],
        buffer_name: str,
    ) -> float:
        try:
            values = data["buffer"][buffer_name]["buffer"]
        except KeyError as exc:
            raise PhyphoxClientError(
                f"phyphox response is missing buffer '{buffer_name}'. "
                "Open the acceleration experiment in phyphox and enable remote access."
            ) from exc

        if not values:
            raise PhyphoxClientError(
                f"phyphox buffer '{buffer_name}' is empty. "
                "Wait for sensor samples to arrive, then try again."
            )

        try:
            return float(values[-1])
        except (TypeError, ValueError) as exc:
            raise PhyphoxClientError(
                f"phyphox buffer '{buffer_name}' did not contain a numeric value."
            ) from exc

    def get_acceleration(self) -> tuple[float, float, float]:
        """
        Read the latest x, y, z acceleration values from phyphox.

        Returns
        -------
        tuple[float, float, float]
            Latest acceleration values for x, y, z.
        """
        url = f"{self.base_url}/get?accX&accY&accZ"

        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise PhyphoxClientError(
                f"Timed out while connecting to phyphox at {self.base_url}. "
                f"{PHYPHOX_NETWORK_HINT}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise PhyphoxClientError(
                f"Could not connect to phyphox at {self.base_url}. "
                f"{PHYPHOX_NETWORK_HINT}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise PhyphoxClientError(
                f"phyphox returned HTTP error {response.status_code}. "
                "Confirm that the URL is copied exactly from phyphox remote access."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise PhyphoxClientError(
                f"Could not read from phyphox at {self.base_url}: {exc}. "
                f"{PHYPHOX_NETWORK_HINT}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise PhyphoxClientError(
                "phyphox did not return valid JSON. "
                "Confirm that the URL points to the phyphox remote access server."
            ) from exc

        x = self._read_latest_buffer_value(data, "accX")
        y = self._read_latest_buffer_value(data, "accY")
        z = self._read_latest_buffer_value(data, "accZ")

        return x, y, z
