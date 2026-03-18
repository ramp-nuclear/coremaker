"""Protocol for isotopic mixtures as far as core creation cares."""

from typing import Any, Container, Iterable, Sequence, Type, TypeVar

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

import numpy as np
from isotopes import ZAID, Isotope, avogadro

from coremaker.materials.util import cumulative_dict, parse_chemical
from coremaker.protocols.mixture import Chemical
from coremaker.protocols.mixture import Mixture as MixtureProtocol

TEMPERATURE_PRECISION = 1e-6


class Mixture(MixtureProtocol):
    """A Concrete implementation of the Mixture Protocol.

    This object represents a homogeneous "gas" of ZAIDs, each with its own number
    density, but where the overall mixture has a single, global, temperature.

    """

    ser_identifier = "Mixture"
    __slots__ = ["isotopes", "temperature", "sab"]

    def __init__(
        self,
        isotopes: dict[ZAID, float],
        temperature: float,
        sab_tables: Iterable[Chemical] = (),
        *,
        ensure_positive: bool = True,
    ):
        """

        Parameters
        ----------
        isotopes: dict[ZAID, float]
            The number density in 1/cm-barn for each ZAID.
        temperature: Degrees Celcius
            The temperature of the ZAID mixture.
        sab_tables: Iterable[Chemical]
            An iterable of chemicals that make up the mixture. Some chemicals
            change the nuclear properties of the isotopes that make up their
            structure, which usually causes a change in the thermal cross-sections.
            This is often referred to as :math:`S_{\alpha\beta}`, hence the name.
            This needs to be accounted for in some materials, such as water.
            A mixture can include more than one chemical.
        ensure_positive: bool
            A flag for whether zero or negative densities should be dropped.
            By default, zero or negative entries are dropped.
        """
        self.isotopes = {iso: nd for iso, nd in isotopes.items() if nd > 0.0 or not ensure_positive}
        self.temperature = temperature
        self.sab: Sequence[Chemical] = tuple(sab_tables)

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, {
            "isotopes": {int(iso): v for iso, v in self.isotopes.items()},
            "temperature": self.temperature,
            "sab_tables": tuple(c.name for c in self.sab),
        }

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *_, **__) -> Self:
        return cls(
            isotopes={Isotope.from_int_with_fallback(int(i)): v for i, v in d["isotopes"].items()},
            temperature=d["temperature"],
            sab_tables=tuple(Chemical[name] for name in d["sab_tables"]),
            ensure_positive=False,
        )

    @classmethod
    def expand(cls, mix: "Mixture", elements: Container[Isotope] = frozenset()) -> "Mixture":
        """Expand elements in the mixture so that they are replaced with
        specific isotopes of that element instead.

        >>> from isotopes import O, H, He4
        >>> mmix = Mixture({H: 2., O: 1., He4: 4.}, 20.)
        >>> nmix = Mixture.expand(mmix)
        >>> list(nmix.isotopes.keys())
        [H1, H2, O16, O17, O18, He4]

        Parameters
        ----------
        mix: Mixture
            The mixture to expand.
        elements: Container[Isotope]
            Specific elements to expand. If False, all elements are expanded on
            sight.

        """
        if elements:

            def _iselem(x):
                return x in elements
        else:

            def _iselem(x):
                return isinstance(x, Isotope) and x.A == 0

        return cls(
            cumulative_dict(
                [
                    (subkey, value * mult)
                    for key, value in mix.items()
                    # Safe to ignore type because of the _iselem check.
                    for subkey, mult in (
                        key.abundance.items()  # type: ignore
                        if _iselem(key)
                        else [(key, 1.0)]
                    )
                ]
            ),
            mix.temperature,
            mix.sab,
        )

    @classmethod
    def by_chemical_formula(
        cls, composition: dict[ZAID | str, float], temperature: float, sab_tables: Iterable[Chemical] = ()
    ) -> "Mixture":
        """Define mixture by a combination of elemental chemical formulas and isotopes.

        Examples
        --------
        >>> from coremaker.materials import Mixture
        >>> from isotopes import H1
        >>> mmix = Mixture.by_chemical_formula({H1: 1., 'H2O': 1., 'CO2': 1.}, 20.)
        >>> mmix.isotopes
        {H1: 1.0, H: 2.0, O: 3.0, C: 1.0}
        >>> smix = Mixture.by_chemical_formula({'Ca2(CO3)3': 1.}, 20.)
        >>> smix.isotopes
        {Ca: 2.0, C: 3.0, O: 9.0}

        Parameters
        ----------
        composition: dict[Isotope | str, float]
            The number concentration of each isotope or compound (given in a string format).
            Given in 1/barn-cm.
        temperature: Degrees Celcius
            The temperature of the ZAID mixture.
        sab_tables: Iterable[Chemical]
            An iterable of chemicals that make up the mixture. Some chemicals
            change the nuclear properties of the isotopes that make up their
            structure, which usually causes a change in the thermal cross-sections.
            This is often referred to as :math:`S_{\alpha\beta}`, hence the name.
            This needs to be accounted for in some materials, such as water.
            A mixture can include more than one chemical, though not all transport
            codes know how to deal with multiple :math:`S_{\alpha\beta}` treatments.

        """
        return cls(
            cumulative_dict(
                [
                    (subkey, n * mult)
                    for key, n in composition.items()
                    for subkey, mult in (parse_chemical(key).items() if isinstance(key, str) else [(key, 1)])
                ]
            ),
            temperature,
            sab_tables,
        )

    @classmethod
    def by_weight_density(
        cls, composition: dict[Isotope, float], temperature: float, sab_tables: Iterable[Chemical] = ()
    ) -> "Mixture":
        """Create a mixture from a bunch of weight densities.

        Parameters
        ----------
        composition - Weight density of each component.
        temperature - Temperature of the mixture.
        sab_tables - S_{αβ} tables to associate with the mixture.

        """
        total_density = sum(composition.values())
        return cls.by_weight_fraction(
            {iso: density / total_density for iso, density in composition.items()},
            total_density,
            temperature,
            sab_tables,
        )

    @classmethod
    def by_weight_fraction(
        cls, composition: dict[Isotope, float], density: float, temperature: float, sab_tables: Iterable[Chemical] = ()
    ) -> "Mixture":
        """Create a mixture given the weight fractions of its components.

        Parameters
        ----------
        composition: dict[Isotope, float]
            The weight fraction of each isotope, as fractions of the density.
        density: g/cc
            The total material density, in g/cc.
        temperature: Degrees Celsius
            Mixture temperature, in degrees Celsius.
        sab_tables: Iterable[Chemical]
            S_{αβ} tables to associate with the mixture.

        """
        nds = {iso: avogadro * density * fraction / iso.mass for iso, fraction in composition.items()}
        return cls(nds, temperature, sab_tables=sab_tables)

    @classmethod
    def alloy(
        cls,
        main: Isotope,
        composition: dict[Isotope, float],
        density: float,
        temperature: float,
        sab_tables: Iterable[Chemical] = (),
    ) -> "Mixture":
        """Make a mixture where there is a dominant component and other weight
        fraction given components.

        Parameters
        ----------
        main: Isotope
            Main ingredient.
        composition: dict[Isotope, float]
            The weight fraction of each isotope.
        density: g/cc
            The total material density, in g/cc.
        temperature: Degrees Celsius
            Mixture temperature, in degrees Celsius.
        sab_tables: Iterable[Chemical]
            S_{αβ} tables to associate with the mixture.

        """
        rest = 1.0 - sum(composition.values())
        c = cumulative_dict([(main, rest), *composition.items()])
        return cls.by_weight_fraction(c, density, temperature, sab_tables=sab_tables)

    @classmethod
    def with_impurities(
        cls, other: "Mixture", impurities: dict[Isotope, float], count: Container[Isotope] = frozenset()
    ) -> "Mixture":
        """Create a new mixture that includes additional impurities.

        Notice that this method increases the total density of the material, as the new impurity does not cause a
        decrease in the original isotopes' densities.

        Parameters
        ----------
        other: Mixture
            Mixture to start with
        impurities: dict[Isotope, float]
            Impurity weight fraction from the counted isotopes.
        count: Container[Isotope]
            Which isotopes from the original mixture to count when computing the impurities weight fraction.

        Returns
        -------
        Mixture
            The new and impure mixture.

        Examples
        --------
        >>> from isotopes import H1, H2, B10
        >>> import numpy as np
        >>> original = Mixture.by_weight_density({H1: 1.}, 20.)
        >>> n = Mixture.with_impurities(original, {H2: 1e-3})
        >>> n.weight_densities()
        {H1: 1.0, H2: 0.001}
        >>> original = Mixture.by_weight_density({H1: 1., H2: 1e-3, B10: 1e-3}, 20.)
        >>> n = Mixture.with_impurities(original, {B10: 1e-3}, count={H1})
        >>> cmpr = {H1: 1., H2: 1e-3, B10: 2e-3}
        >>> assert np.allclose(sorted(cmpr.items(), key=lambda x: x[0]),
        ...                    sorted(n.weight_densities().items(), key=lambda x: x[0]),
        ...                    rtol=1e-5), (cmpr, n)

        """
        count = count or {key for key in other.keys() if isinstance(key, Isotope)}
        new = cls(
            {key: value for key, value in other.items() if key in count},
            other.temperature,
            other.sab,
            ensure_positive=True,
        )
        weight = sum(new.weight_densities().values())
        nds = {iso: frac * weight * avogadro / iso.mass for iso, frac in impurities.items()}
        isos = {iso: other.get(iso, 0.0) + nds.get(iso, 0.0) for iso in set(nds.keys()) | set(other.keys())}
        return cls(isos, other.temperature, other.sab, ensure_positive=True)

    def __eq__(self, other: MixtureProtocol):
        return (
            self.temperature == other.temperature
            and self.isotopes == other.isotopes
            and frozenset(self.sab) == frozenset(other.sab)
        )

    def __hash__(self):
        return hash((frozenset(self.items()), self.temperature, frozenset(self.sab)))

    def __getitem__(self, key: ZAID):
        try:
            return self.isotopes[key]
        except KeyError:
            if key in self.expand(self).isotopes:
                return self.expand(self)[key]
            raise KeyError

    def __contains__(self, iso: ZAID) -> bool:
        """Test if isotope is a key in the mixture.

        Parameters
        ----------
        iso: Isotope to check if in the mixture

        """

        return iso in self.isotopes or iso in self.expand(self).isotopes

    def __iter__(self) -> Iterable[ZAID]:
        yield from self.isotopes

    def keys(self) -> Iterable[ZAID]:
        """Return a keys-view into which isotopes are in the mixture, like
        dict's keys method.
        """
        return self.isotopes.keys()

    def values(self) -> Iterable[float]:
        """Returns the number densities of the isotopes, like dict's values
        method. These are given in 1/cm-barn
        """
        return self.isotopes.values()

    def items(self) -> Iterable[tuple[ZAID, float]]:
        """Returns tuples of key-value for the isotope and number densities,
        like a dict's items method.
        """
        return self.isotopes.items()

    def __repr__(self) -> str:
        return f"Mixture<temperature [degC]: {self.temperature:.1f}, isotopes: {self.isotopes}, S_{{αβ}}: {self.sab}>"

    __str__ = __repr__

    def weight_densities(self) -> dict[ZAID, float]:
        """
        Returns
        -------
        A mapping between each isotope in the mixture and its corresponding
        weight density in g/cc.
        """
        # Safe because we check this is in isotope and they have mass...
        return {
            iso: den * iso.mass / avogadro if isinstance(iso, Isotope) else np.nan  # type: ignore
            for iso, den in self.isotopes.items()
        }


def just_positives(mix: Mixture) -> Mixture:
    """Filters the mixture to only include strictly positive number densities.
    Through burnup or other calculations, a small numeric error could cause
    a zero or negative number density for some ZAIDs.

    Parameters
    ----------
    mix: Mixture
        The original mixture to filter

    Returns
    -------
    Mixture
        A new, filtered mixture.

    """
    return Mixture(
        temperature=mix.temperature,
        isotopes={iso: den for iso, den in mix.isotopes.items() if den > 0.0},
        sab_tables=mix.sab,
    )
