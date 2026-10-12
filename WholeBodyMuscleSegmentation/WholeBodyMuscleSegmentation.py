import os
import subprocess
import tempfile
import logging
import importlib.util
import shutil
import sys
import glob
import re
import time
import threading
import queue
import signal
import csv

import slicer
import vtk, qt, ctk
from slicer.ScriptedLoadableModule import *
import ast
import json
import urllib.request
import colorsys


# Version of the SlicerMuscleMap extension. Bump this and tag the release
# (git tag v<version>) when merging user-visible changes to main.
__version__ = "1.0.0"

MIN_PYTHON_VERSION = (3, 8)

# MuscleMap downloads its models (.pth + .json) from Zenodo into
# <site-packages>/scripts/models/<region>/v<version>/ on first use. The label
# names are read from that json, so they always match the model mm_segment ran.
# The available regions and versions come from ZENODO_MODELS in MuscleMap's
# mm_util.py; this list is only used until MuscleMap is installed.
DEFAULT_REGIONS = ["wholebody", "abdomen", "pelvis", "thigh", "leg", "forearm"]
DEFAULT_REGION = "wholebody"
LATEST_VERSION = "latest"


def labels_json_filename(region: str) -> str:
    return f"contrast_agnostic_{region}_model.json"


class WholeBodyMuscleSegmentation(ScriptedLoadableModule):
    """Main entry point for the MuscleMap whole-body segmentation module."""

    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "MuscleMap Whole-body Segmentation"
        parent.categories = ["MuscleMap"]
        parent.contributors = ["Eddo Wesselink and Kenneth Arnold Weber"]

        parent.helpText = (
            "MuscleMap provides automated muscle segmentation directly inside 3D Slicer.\n\n"
            "Features:\n"
            " • Fully automated whole-body muscle segmentation\n"
            " • Regional models (abdomen, pelvis, thigh, leg, forearm) and selectable model versions\n"
            " • Support for CT and MRI\n"
            " • Export of labelled segmentations (NIfTI labelmap, CSV label table, color table)\n\n"
            "MuscleMap requires the SlicerPyTorch extension. After installation, all dependencies\n"
            "can be installed using the 'Install MuscleMap dependencies' button.\n\n"
            f"SlicerMuscleMap version: {__version__}\n\n"
            "For documentation and source code, visit:\n"
            "https://github.com/MuscleMap/MuscleMap"
        )

        parent.acknowledgementText = (
            "MuscleMap is developed by the MuscleMap Consortium.\n\n"
            "If you use this extension in academic or scientific work, please cite the following publications:\n\n"
            "------------------------------------------------------------\n\n"
            "1) McKay MJ, Weber KA 2nd, Wesselink EO, Smith ZA, Abbott R, Anderson DB, Ashton-James CE, "
            "Atyeo J, Beach AJ, Burns J, Clarke S, Collins NJ, Coppieters MW, Cornwall J, Crawford RJ, "
            "De Martino E, Dunn AG, Eyles JP, Feng HJ, Fortin M, Franettovich Smith MM, Galloway G, "
            "Gandomkar Z, Glastras S, Henderson LA, Hides JA, Hiller CE, Hilmer SN, Hoggarth MA, Kim B, "
            "Lal N, LaPorta L, Magnussen JS, Maloney S, March L, Nackley AG, O'Leary SP, Peolsson A, "
            "Perraton Z, Pool-Goudzwaard AL, Schnitzler M, Seitz AL, Semciw AI, Sheard PW, Smith AC, "
            "Snodgrass SJ, Sullivan J, Tran V, Valentin S, Walton DM, Wishart LR, Elliott JM. "
            "MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. "
            "J Imaging. 2024;10(11):262. https://doi.org/10.3390/jimaging10110262\n\n"
            "------------------------------------------------------------\n\n"
            "2) Wesselink EO, Elliott JM, Coppieters MW, Hancock MJ, Cronin B, Pool-Goudzwaard A, "
            "Weber II KA. Convolutional neural networks for the automatic segmentation of lumbar "
            "paraspinal muscles in people with low back pain. Sci Rep. 2022;12(1):13485. "
            "https://doi.org/10.1038/s41598-022-16710-5\n\n"
            "------------------------------------------------------------\n\n"
            "MuscleMap is built upon open-source software including:\n"
            " • 3D Slicer (https://slicer.org)\n"
            " • MONAI (https://monai.io)\n"
            " • PyTorch (https://pytorch.org)\n"
            " • Nibabel\n\n"
            "We gratefully acknowledge the developers and contributors of these projects."
        )

        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "MuscleMap.png")
        parent.icon = qt.QIcon(iconPath)

    def icon(self):
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "MuscleMap.png")
        return qt.QIcon(iconPath)


class WholeBodyMuscleSegmentationWidget(ScriptedLoadableModuleWidget):
    """User interface for the MuscleMap segmentation module."""

    SETTINGS_PREFIX = "MuscleMap/"

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = WholeBodyMuscleSegmentationLogic()

        logoLabel = qt.QLabel()
        logoPath = os.path.join(os.path.dirname(__file__), "MuscleMap.png")
        pixmap = qt.QPixmap(logoPath)
        scaled = pixmap.scaledToWidth(200, qt.Qt.SmoothTransformation)
        logoLabel.setPixmap(scaled)
        logoLabel.setAlignment(qt.Qt.AlignCenter)
        self.layout.addWidget(logoLabel)

        packagesCollapsibleButton = ctk.ctkCollapsibleButton()
        packagesCollapsibleButton.text = "Installing packages"
        self.layout.addWidget(packagesCollapsibleButton)

        packagesLayout = qt.QVBoxLayout(packagesCollapsibleButton)

        self.extensionsButton = qt.QPushButton("Prerequisite: Complete this step before proceeding")
        self.extensionsButton.toolTip = "Shows instructions for installing required python packages (e.g. PyTorch)."
        self.extensionsButton.connect("clicked()", self.onExtensionsClicked)
        packagesLayout.addWidget(self.extensionsButton)

        self.installButton = qt.QPushButton("Install MuscleMap dependencies")
        self.installButton.toolTip = "Install the MuscleMap toolbox into Slicer's Python using pip."
        self.installButton.connect("clicked()", self.onInstallClicked)
        packagesLayout.addWidget(self.installButton)

        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Inputs"
        self.layout.addWidget(parametersCollapsibleButton)

        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.loadFromFileButton = qt.QPushButton("Load volume from file...")
        self.loadFromFileButton.toolTip = "Select an image file from disk and load it as input volume."
        self.loadFromFileButton.connect("clicked()", self.onLoadFromFileClicked)
        parametersFormLayout.addRow(self.loadFromFileButton)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = True
        self.inputSelector.noneDisplay = ""
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Select the input image.")
        parametersFormLayout.addRow("Input volume:", self.inputSelector)

        self.regionComboBox = qt.QComboBox()
        self.regionComboBox.setToolTip(
            "Anatomical region model to segment with (mm_segment -r). "
            "The model is downloaded from Zenodo and cached on first use."
        )
        parametersFormLayout.addRow("Model:", self.regionComboBox)

        self.versionComboBox = qt.QComboBox()
        self.versionComboBox.setToolTip(
            "Model version to use (mm_segment --model_version). "
            "'latest' always uses the newest version available on Zenodo."
        )
        parametersFormLayout.addRow("Model version:", self.versionComboBox)

        advancedCollapsibleButton = ctk.ctkCollapsibleButton()
        advancedCollapsibleButton.text = "Advanced"
        advancedCollapsibleButton.collapsed = True
        self.layout.addWidget(advancedCollapsibleButton)

        advancedLayout = qt.QFormLayout(advancedCollapsibleButton)

        self.forceCpuCheckBox = qt.QCheckBox("Force CPU (ignore GPU)")
        self.forceCpuCheckBox.setToolTip("If checked, MuscleMap will run on the CPU even if a GPU is available.")
        self.forceCpuCheckBox.checked = False
        advancedLayout.addRow(self.forceCpuCheckBox)

        self.forceLowerOverlapCheckBox = qt.QCheckBox("Force lower overlap")
        self.forceLowerOverlapCheckBox.setToolTip("If checked, run mm_segment with -s 75. Otherwise uses -s 90.")
        self.forceLowerOverlapCheckBox.checked = False
        advancedLayout.addRow(self.forceLowerOverlapCheckBox)

        self.rememberSettingsCheckBox = qt.QCheckBox("Remember settings")
        self.rememberSettingsCheckBox.setToolTip(
            "If checked, the selected model, model version and advanced options "
            "are restored the next time Slicer is started."
        )
        self.rememberSettingsCheckBox.checked = False
        advancedLayout.addRow(self.rememberSettingsCheckBox)

        self.runButton = qt.QPushButton("Run MuscleMap segmentation")
        self.runButton.toolTip = "Run mm_segment on the selected volume."
        self.runButton.connect("clicked()", self.onRunClicked)
        self.layout.addWidget(self.runButton)

        outputsCollapsibleButton = ctk.ctkCollapsibleButton()
        outputsCollapsibleButton.text = "Outputs"
        self.layout.addWidget(outputsCollapsibleButton)

        outputsLayout = qt.QFormLayout(outputsCollapsibleButton)

        self.outputSegmentationSelector = slicer.qMRMLNodeComboBox()
        self.outputSegmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSegmentationSelector.selectNodeUponCreation = False
        self.outputSegmentationSelector.addEnabled = True
        self.outputSegmentationSelector.removeEnabled = True
        self.outputSegmentationSelector.noneEnabled = True
        self.outputSegmentationSelector.showHidden = False
        self.outputSegmentationSelector.showChildNodeTypes = False
        self.outputSegmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSegmentationSelector.setToolTip("Select the segmentation for 3D visualization and export.")
        outputsLayout.addRow("Segmentation:", self.outputSegmentationSelector)

        self.show3DButton = slicer.qMRMLSegmentationShow3DButton()
        self.show3DButton.setToolTip("Show 3D representation of the selected segmentation.")
        outputsLayout.addRow(self.show3DButton)

        self.exportButton = qt.QPushButton("Export segmentation...")
        self.exportButton.toolTip = (
            "Save the selected segmentation as a NIfTI labelmap with the original MuscleMap "
            "label values, together with a CSV label table and a Slicer color table."
        )
        self.exportButton.connect("clicked()", self.onExportClicked)
        outputsLayout.addRow(self.exportButton)

        self.outputSegmentationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.show3DButton.setSegmentationNode
        )

        self.layout.addStretch(1)

        self.regionComboBox.connect("currentIndexChanged(int)", self.updateVersionComboBox)
        self.updateModelComboBoxes()
        self.restoreSettings()

        self.regionComboBox.connect("currentIndexChanged(int)", self.saveSettings)
        self.versionComboBox.connect("currentIndexChanged(int)", self.saveSettings)
        self.forceCpuCheckBox.connect("toggled(bool)", self.saveSettings)
        self.forceLowerOverlapCheckBox.connect("toggled(bool)", self.saveSettings)
        self.rememberSettingsCheckBox.connect("toggled(bool)", self.saveSettings)

    # --- settings -------------------------------------------------------

    def restoreSettings(self):
        settings = qt.QSettings()
        remember = settings.value(self.SETTINGS_PREFIX + "rememberSettings", False)
        remember = str(remember).lower() in ("true", "1")
        self.rememberSettingsCheckBox.checked = remember
        if not remember:
            return

        region = settings.value(self.SETTINGS_PREFIX + "region", DEFAULT_REGION)
        index = self.regionComboBox.findText(str(region))
        if index >= 0:
            self.regionComboBox.setCurrentIndex(index)

        version = settings.value(self.SETTINGS_PREFIX + "modelVersion", LATEST_VERSION)
        index = self.versionComboBox.findText(str(version))
        if index >= 0:
            self.versionComboBox.setCurrentIndex(index)

        for key, checkBox in (("forceCpu", self.forceCpuCheckBox),
                              ("forceLowerOverlap", self.forceLowerOverlapCheckBox)):
            value = settings.value(self.SETTINGS_PREFIX + key, False)
            checkBox.checked = str(value).lower() in ("true", "1")

    def saveSettings(self, _value=None):
        settings = qt.QSettings()
        remember = bool(self.rememberSettingsCheckBox.isChecked())
        settings.setValue(self.SETTINGS_PREFIX + "rememberSettings", remember)
        if not remember:
            for key in ("region", "modelVersion", "forceCpu", "forceLowerOverlap"):
                settings.remove(self.SETTINGS_PREFIX + key)
            return
        settings.setValue(self.SETTINGS_PREFIX + "region", self.regionComboBox.currentText)
        settings.setValue(self.SETTINGS_PREFIX + "modelVersion", self.versionComboBox.currentText)
        settings.setValue(self.SETTINGS_PREFIX + "forceCpu", bool(self.forceCpuCheckBox.isChecked()))
        settings.setValue(self.SETTINGS_PREFIX + "forceLowerOverlap", bool(self.forceLowerOverlapCheckBox.isChecked()))

    # --- model selection ------------------------------------------------

    def updateModelComboBoxes(self):
        """(Re)fill the region list from the installed MuscleMap package, keeping the current choice."""
        self.availableModels = self.logic.getAvailableModels()
        current = self.regionComboBox.currentText or DEFAULT_REGION

        wasBlocked = self.regionComboBox.blockSignals(True)
        self.regionComboBox.clear()
        for region in self.availableModels:
            self.regionComboBox.addItem(region)
        index = self.regionComboBox.findText(current)
        self.regionComboBox.setCurrentIndex(index if index >= 0 else 0)
        self.regionComboBox.blockSignals(wasBlocked)

        self.updateVersionComboBox()

    def updateVersionComboBox(self, _index=None):
        versions = self.availableModels.get(self.regionComboBox.currentText, [])
        current = self.versionComboBox.currentText

        self.versionComboBox.clear()
        self.versionComboBox.addItem(LATEST_VERSION)
        for version in versions:
            self.versionComboBox.addItem(version)
        index = self.versionComboBox.findText(current)
        self.versionComboBox.setCurrentIndex(index if index >= 0 else 0)
        # Regions without published versions only offer 'latest'
        self.versionComboBox.setEnabled(bool(versions))

    # --- actions --------------------------------------------------------

    def onLoadFromFileClicked(self):
        fileFilters = (
            "Volume files (*.nii *.nii.gz *.nrrd *.mha *.mhd);;"
            "All files (*)"
        )
        filePath = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select input volume",
            "",
            fileFilters
        )
        if not filePath:
            return

        volumeNode = slicer.util.loadVolume(filePath)
        if not volumeNode:
            slicer.util.errorDisplay(f"Failed to load volume from file:\n{filePath}")
            return

        self.inputSelector.setCurrentNode(volumeNode)

    def onExtensionsClicked(self):
        msg = (
            "MuscleMap requires the PyTorch package to be installed in 3D Slicer via the SlicerPyTorch extension.\n\n"
            "Step 1 – Install the 'PyTorch' (SlicerPyTorch) extension:\n"
            "  • In Slicer, go to:  View → Extensions Manager\n"
            "  • Search for:  PyTorch or SlicerPyTorch\n"
            "  • Install the extension and restart Slicer.\n\n"
            "Step 2 – Download and install PyTorch inside the SlicerPyTorch module:\n"
            "  • Go to:  View → Modules\n"
            "  • Choose:  Utilities → PyTorch (or 'SlicerPyTorch')\n"
            "  • Click the button 'Install PyTorch' and select the CPU or GPU build that matches your system.\n\n"
            "Step 3 – Return to the MuscleMap module and click 'Install MuscleMap dependencies'\n"
            "         to install the remaining Python packages used by MuscleMap.\n\n"
            "We recommend installing the appropriate PyTorch wheel using pip as suggested by the SlicerPyTorch module."
        )
        slicer.util.infoDisplay(msg, windowTitle="Prerequisite: Complete this step before proceeding")

    def onInstallClicked(self):
        progress = qt.QProgressDialog(
            "Installing MuscleMap dependencies...\n\n"
            "This may take a while, depending on your internet connection.",
            None, 0, 0, slicer.util.mainWindow()
        )
        progress.windowTitle = "Installing MuscleMap"
        progress.setWindowModality(qt.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        slicer.app.processEvents()

        try:
            self.logic.ensureDependencies()
            self.updateModelComboBoxes()
            progress.close()
            slicer.util.infoDisplay("MuscleMap dependencies are installed and ready.")
        except Exception as e:
            progress.close()
            slicer.util.errorDisplay(f"Failed to install/check dependencies:\n{e}")

    def onRunClicked(self):
        inputVolume = self.inputSelector.currentNode()
        if not inputVolume:
            slicer.util.errorDisplay("Please select an input volume first.")
            return

        force_cpu = bool(self.forceCpuCheckBox.isChecked())
        force_lower_overlap = bool(self.forceLowerOverlapCheckBox.isChecked())
        region = self.regionComboBox.currentText or DEFAULT_REGION
        model_version = self.versionComboBox.currentText or LATEST_VERSION

        try:
            import torch
            if force_cpu:
                device = "CPU (forced)"
            else:
                device = "GPU" if torch.cuda.is_available() else "CPU"
        except Exception:
            device = "CPU"

        header = (
            f"MuscleMap {region} segmentation (model version: {model_version}, extension {__version__})\n"
            f"Computation device: {device}\n"
        )
        if device.startswith("CPU"):
            header += (
                "\nNote: processing on the CPU can be slow. Enabling 'Force lower overlap' "
                "under Advanced reduces the processing time.\n"
            )
        header += (
            "\nIf you use MuscleMap for research, please cite the MuscleMap publications "
            "(see module Acknowledgements).\n"
        )
        logging.info(f"[MuscleMap] {header.replace(os.linesep, ' ')}")

        progress = qt.QProgressDialog("", "Cancel", 0, 0, slicer.util.mainWindow())
        progress.windowTitle = "MuscleMap segmentation"
        progress.setWindowModality(qt.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(500)

        def setStatus(status):
            progress.labelText = header + "\n" + status
            slicer.app.processEvents()

        setStatus("Preparing the model. If it is not yet available locally, it is downloaded from Zenodo. "
                  "This may take a while.")
        progress.show()
        slicer.app.processEvents()

        self.runButton.enabled = False
        try:
            segNode = self.logic.runSegmentation(
                inputVolume,
                force_cpu=force_cpu,
                force_lower_overlap=force_lower_overlap,
                region=region,
                model_version=model_version,
                statusCallback=setStatus,
                cancelRequested=lambda: progress.wasCanceled,
            )
            progress.close()
            if segNode:
                self.outputSegmentationSelector.setCurrentNode(segNode)
                self.show3DButton.setSegmentationNode(segNode)
                slicer.util.infoDisplay(f"MuscleMap {region} segmentation completed.")
        except SegmentationCancelled:
            progress.close()
            slicer.util.infoDisplay("MuscleMap segmentation was cancelled.")
        except Exception as e:
            progress.close()
            slicer.util.errorDisplay(f"Segmentation failed:\n{e}")
        finally:
            progress.close()
            self.runButton.enabled = True

    def onExportClicked(self):
        segmentationNode = self.outputSegmentationSelector.currentNode()
        if not segmentationNode:
            slicer.util.errorDisplay("Please select a segmentation to export first.")
            return

        filePath = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Export segmentation",
            self.logic.defaultExportPath(segmentationNode),
            "NIfTI labelmap (*.nii.gz)",
        )
        if not filePath:
            return

        try:
            written = self.logic.exportSegmentation(segmentationNode, filePath)
        except Exception as e:
            slicer.util.errorDisplay(f"Export failed:\n{e}")
            return

        msg = "Exported:\n" + "\n".join(f"  {p}" for p in written)
        if not segmentationNode.GetNodeReference(self.logic.COLOR_TABLE_REFERENCE_ROLE):
            msg += (
                "\n\nNote: this segmentation was not created by MuscleMap, so the label values "
                "in the labelmap are numbered consecutively."
            )
        slicer.util.infoDisplay(msg, windowTitle="MuscleMap export")


class SegmentationCancelled(Exception):
    """Raised when the user cancels a running segmentation."""


class WholeBodyMuscleSegmentationLogic(ScriptedLoadableModuleLogic):
    """Processing code for the MuscleMap segmentation module."""

    # Node reference from a MuscleMap segmentation to its color table (label values + names)
    COLOR_TABLE_REFERENCE_ROLE = "MuscleMapColorTable"
    # Segmentation node attribute holding the model's label list as JSON
    LABELS_ATTRIBUTE = "MuscleMap.Labels"

    def _fetch_json(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": f"SlicerMuscleMap/{__version__}"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _version_key(version: str):
        try:
            return tuple(int(x) for x in version.split("."))
        except ValueError:
            return ()

    def _musclemap_package_dirs(self):
        """Directories of the installed MuscleMap package ('scripts'), without importing it."""
        spec = importlib.util.find_spec("scripts")
        if spec is None or not spec.submodule_search_locations:
            return []
        return [d for d in spec.submodule_search_locations if os.path.isfile(os.path.join(d, "mm_util.py"))]

    def _musclemap_supports_regions(self) -> bool:
        """True if the installed MuscleMap downloads region models from Zenodo (has ZENODO_MODELS)."""
        for pkgDir in self._musclemap_package_dirs():
            try:
                with open(os.path.join(pkgDir, "mm_util.py"), encoding="utf-8") as f:
                    if "ZENODO_MODELS" in f.read():
                        return True
            except OSError:
                continue
        return False

    def _read_zenodo_models(self) -> dict:
        """Read the ZENODO_MODELS table from the installed mm_util.py.

        mm_util imports torch/monai, so it is parsed instead of imported. Returns
        {} if MuscleMap is not installed, predates the Zenodo model download, or
        the table cannot be parsed (the UI then falls back to DEFAULT_REGIONS).
        """
        for pkgDir in self._musclemap_package_dirs():
            try:
                with open(os.path.join(pkgDir, "mm_util.py"), encoding="utf-8") as f:
                    tree = ast.parse(f.read())
            except (OSError, SyntaxError) as e:
                logging.warning(f"[MuscleMap] Could not parse mm_util.py: {e}")
                continue
            for node in tree.body:
                if isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                elif isinstance(node, ast.Assign):
                    targets = node.targets
                else:
                    continue
                if any(isinstance(t, ast.Name) and t.id == "ZENODO_MODELS" for t in targets):
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        return {}
        return {}

    def getAvailableModels(self) -> dict:
        """Return {region: [versions, newest first]} that mm_segment can download."""
        zenodo_models = self._read_zenodo_models()
        if not zenodo_models:
            return {region: [] for region in DEFAULT_REGIONS}
        regions = sorted(zenodo_models, key=lambda r: (r != DEFAULT_REGION, r))
        return {
            region: sorted(zenodo_models[region].get("versions", {}), key=self._version_key, reverse=True)
            for region in regions
        }

    def _find_cached_labels_json(self, region: str, model_version: str = LATEST_VERSION, mm_segment_log: str = ""):
        """Locate the model json for *region* in MuscleMap's local model cache.

        Prefers the version mm_segment reported in its log, then the requested
        version, otherwise the highest cached version. Returns None if nothing is cached.
        """
        jsonFilename = labels_json_filename(region)
        candidates = {}  # version folder name -> path
        for pkgDir in self._musclemap_package_dirs():
            regionDir = os.path.join(pkgDir, "models", region)
            for path in glob.glob(os.path.join(regionDir, "v*", jsonFilename)):
                candidates[os.path.basename(os.path.dirname(path))] = path
            # Pre-Zenodo MuscleMap installs shipped the json without a version folder
            legacy = os.path.join(regionDir, jsonFilename)
            if os.path.isfile(legacy):
                candidates.setdefault("", legacy)
        if not candidates:
            return None

        m = re.search(rf"'{re.escape(region)}' model v(\d+(?:\.\d+)*)", mm_segment_log)
        for wanted in ([m.group(1)] if m else []) + [model_version]:
            if f"v{wanted}" in candidates:
                return candidates[f"v{wanted}"]

        return candidates[max(candidates, key=lambda name: self._version_key(name[1:]))]

    def _load_labels_json(self, region: str, model_version: str = LATEST_VERSION, mm_segment_log: str = "") -> dict:
        path = self._find_cached_labels_json(region, model_version, mm_segment_log)
        if path:
            logging.info(f"[MuscleMap] Reading label names from {path}")
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        # Fallback: fetch the json directly from Zenodo
        model_info = self._read_zenodo_models().get(region)
        if not model_info:
            raise RuntimeError(f"No cached model json and no Zenodo record known for region '{region}'.")
        logging.info(f"[MuscleMap] No cached model json found, fetching '{region}' label names from Zenodo...")
        version_record_id = model_info.get("versions", {}).get(model_version)
        if version_record_id:
            url = f"https://zenodo.org/api/records/{version_record_id}"
        else:
            url = f"https://zenodo.org/api/records/{model_info['record_id']}/versions/latest"
        jsonFilename = labels_json_filename(region)
        for f in self._fetch_json(url).get("files", []):
            if f.get("key") == jsonFilename:
                return self._fetch_json(f["links"]["self"])
        raise RuntimeError(f"'{jsonFilename}' not found in the Zenodo record.")

    def _build_colortable_from_labels_json(self, region: str, model_version: str = LATEST_VERSION, mm_segment_log: str = ""):
        data = self._load_labels_json(region, model_version, mm_segment_log)
        labels = data.get("labels", [])
        if not labels:
            raise RuntimeError(f"No 'labels' found in {region} model json.")

        labels_sorted = sorted(labels, key=lambda x: int(x["value"]))
        max_value = max(int(x["value"]) for x in labels_sorted)

        colorNodeName = f"MuscleMap_{region}_ColorTable"

        try:
            colorNode = slicer.util.getNode(colorNodeName)
            if not colorNode.IsA("vtkMRMLColorTableNode"):
                raise slicer.util.MRMLNodeNotFoundException()
        except Exception:
            colorNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLColorTableNode", colorNodeName
            )
            colorNode.SetTypeToUser()

        colorNode.SetNumberOfColors(max_value + 1)
        # A reused table may still hold entries of another model version
        for value in range(max_value + 1):
            colorNode.SetColor(value, "", 0.0, 0.0, 0.0, 0.0)

        # Background
        colorNode.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)

        base_colors = {}  # anatomy -> hue (float 0..1)

        base_saturation_left = 0.85
        base_value_left = 0.65

        saturation_right = 0.45
        value_right = 0.95

        for item in labels_sorted:
            anatomy = (item.get("anatomy") or "").strip()
            side_raw = (item.get("side") or "").strip()
            side = side_raw.lower()
            label_value = int(item["value"])

            if anatomy not in base_colors:
                h = (len(base_colors) * 0.61803398875) % 1.0
                base_colors[anatomy] = h

            h = base_colors[anatomy]

            if side == "right":
                s = saturation_right
                v = value_right
            else:
                s = base_saturation_left
                v = base_value_left

            r, g, b = colorsys.hsv_to_rgb(h, s, v)

            if side_raw and side != "no side":
                name = f"{anatomy} ({side_raw})"
            else:
                name = anatomy

            colorNode.SetColor(label_value, name, float(r), float(g), float(b), 1.0)

        return colorNode, labels_sorted

    def ensureDependencies(self):
        if sys.version_info < MIN_PYTHON_VERSION:
            msg = (
                "MuscleMap requires at least Python "
                f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}.\n\n"
                f"You are currently running Python {sys.version_info.major}.{sys.version_info.minor}.\n\n"
                "Please install a more recent 3D Slicer version "
                "(for example Slicer 5.2 or newer) from https://slicer.org "
                "and then reinstall the MuscleMap extension."
            )
            logging.error("[MuscleMap] " + msg.replace("\n", " "))
            try:
                slicer.util.errorDisplay(msg, windowTitle="MuscleMap – incompatible Python")
            except Exception:
                pass
            raise RuntimeError(msg)

        def have_module(name: str) -> bool:
            return importlib.util.find_spec(name) is not None

        have_monai   = have_module("monai")
        have_nibabel = have_module("nibabel")
        have_pandas  = have_module("pandas")
        have_torch   = have_module("torch")
        have_sklearn = have_module("sklearn")
        have_scipy   = have_module("scipy")
        have_tqdm    = have_module("tqdm")
        have_psutil  = have_module("psutil")

        if not have_torch:
            logging.error("[MuscleMap] PyTorch (torch) is not accessible in Slicer.")
            raise RuntimeError(
                "PyTorch (torch) is not installed in this 3D Slicer Python environment.\n\n"
                "To install PyTorch correctly, please follow these steps:\n\n"
                "  1) Install the 'PyTorch' / 'SlicerPyTorch' extension:\n"
                "       • In Slicer, go to:  View → Extensions Manager\n"
                "       • Search for:  PyTorch or SlicerPyTorch\n"
                "       • Install the extension and restart Slicer\n\n"
                "  2) After restarting, open the SlicerPyTorch utility module:\n"
                "       • Go to:  View → Modules\n"
                "       • Choose:  Utilities → PyTorch (or 'SlicerPyTorch')\n\n"
                "  3) In that module, click the button:\n"
                "       “Install PyTorch”  (CPU or GPU version depending on your system)\n\n"
                "After completing these steps, return to MuscleMap and try again."
            )

        import torch  # noqa: F401
        # Look for the MuscleMap package in Slicer's Python rather than for the
        # `mm_segment` entry point on PATH: PATH may hold another installation.
        # Versions from before the Zenodo model download do not support region
        # models / --model_version; treat those as missing so they get upgraded.
        have_musclemap = bool(self._musclemap_package_dirs())
        if have_musclemap and not self._musclemap_supports_regions():
            logging.info("[MuscleMap] Installed MuscleMap package is outdated, it will be upgraded.")
            have_musclemap = False

        if all([
            have_monai, have_nibabel, have_torch, have_sklearn, have_scipy,
            have_tqdm, have_psutil, have_pandas, have_musclemap
        ]):
            logging.info("[MuscleMap] All dependencies already present, nothing to install.")
            return

        logging.info("[MuscleMap] Missing dependencies detected, installing...")

        # torch (provided by the SlicerPyTorch extension) and numpy (bundled by
        # Slicer and ABI-critical for its C++ side) must never be installed or
        # upgraded by pip, or the Slicer Python environment breaks. We pin them to
        # their currently installed versions via a pip constraints file, so that
        # even a transitive dependency cannot silently change them - pip fails
        # loudly instead of quietly breaking Slicer.
        import numpy
        constraints = [
            f"numpy=={numpy.__version__}",
            f"torch=={torch.__version__}",
        ]
        constraintsPath = os.path.join(tempfile.gettempdir(), "musclemap_pip_constraints.txt")
        with open(constraintsPath, "w") as f:
            f.write("\n".join(constraints) + "\n")
        logging.info(f"[MuscleMap] Using pip constraints {constraints} ({constraintsPath})")

        # Runtime dependencies of `mm_segment` - the only MuscleMap tool the Slicer
        # extension runs. Derived from the imports of scripts/mm_segment.py and
        # scripts/mm_util.py. The GUI/metrics-only packages from the MuscleMap
        # requirements.txt (scikit-image, matplotlib, customtkinter, Pillow,
        # transforms3d) are intentionally NOT installed: the segmentation path
        # never imports them.
        #
        # Intentionally UNPINNED. A modern Slicer already ships a recent scientific
        # stack (numpy, scipy, scikit-learn, pandas, ...); forcing MuscleMap's old
        # requirements.txt pins (e.g. scikit-learn==1.3.2) would downgrade those
        # and break Slicer - scikit-learn 1.3.2 cannot import against numpy 2.x.
        # Installing unpinned, WITHOUT --no-deps, keeps whatever Slicer already
        # has, installs only what is missing (e.g. monai, psutil), and pulls each
        # package's missing transitive deps - most notably `narwhals`, which modern
        # scikit-learn imports at startup.
        segmentation_packages = [
            "monai",
            "nibabel",
            "scikit-learn",
            "scipy",
            "pandas",
            "tqdm",
            "psutil",
        ]

        for pkg in segmentation_packages:
            logging.info(f"[MuscleMap] Installing (constrained, with deps): {pkg}")
            slicer.util.pip_install(["-c", constraintsPath, pkg])

        # The MuscleMap package itself is installed WITHOUT its declared
        # dependencies: its requirements.txt pins torch==2.4.1 and numpy==1.24.4,
        # which would clobber the Slicer-managed builds. We only need its
        # `mm_segment` entry point here; everything mm_segment imports is provided
        # by segmentation_packages above.
        if not have_musclemap:
            logging.info("[MuscleMap] Installing MuscleMap package (no deps)...")
            # --force-reinstall: an outdated install may carry the same package version
            slicer.util.pip_install([
                "--no-deps", "--upgrade", "--force-reinstall",
                "git+https://github.com/MuscleMap/MuscleMap.git",
            ])
        else:
            logging.info("[MuscleMap] MuscleMap package already installed, skipping.")

        logging.info("[MuscleMap] Dependency check/installation finished.")

    def _killProcessTree(self, proc):
        """Stop mm_segment. sys.executable is the Slicer launcher, so its child must go too."""
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as e:
            logging.warning(f"[MuscleMap] Could not stop mm_segment process tree: {e}")
            proc.kill()
        proc.wait()

    def _runProcess(self, cmd, cwd, env, statusCallback=None, cancelRequested=None) -> str:
        """Run *cmd* without blocking the Slicer UI; returns the combined output.

        Lines are logged as they arrive, the status callback is updated on
        model download / inference start, and the process is stopped when
        cancelRequested() returns True.
        """
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=env,
            startupinfo=startupinfo,
            start_new_session=(os.name != "nt"),
        )

        lines = queue.Queue()

        def reader():
            for line in iter(proc.stdout.readline, ""):
                lines.put(line)
            proc.stdout.close()

        readerThread = threading.Thread(target=reader, daemon=True)
        readerThread.start()

        output = []

        def drain():
            while True:
                try:
                    line = lines.get_nowait()
                except queue.Empty:
                    return
                output.append(line)
                logging.info("[mm_segment] " + line.rstrip())
                if not statusCallback:
                    continue
                if "Downloading" in line and "Zenodo" in line:
                    statusCallback("Downloading the model from Zenodo. This may take a while.")
                elif "Task: Segmentation" in line:
                    statusCallback("Running the segmentation. This may take several minutes.")

        while proc.poll() is None:
            drain()
            if cancelRequested and cancelRequested():
                logging.info("[MuscleMap] Segmentation cancelled by user, stopping mm_segment.")
                self._killProcessTree(proc)
                raise SegmentationCancelled()
            slicer.app.processEvents()
            time.sleep(0.1)

        readerThread.join(timeout=5.0)
        drain()
        return "".join(output)

    def runSegmentation(
        self,
        inputVolumeNode,
        force_cpu: bool = False,
        force_lower_overlap: bool = False,
        region: str = DEFAULT_REGION,
        model_version: str = LATEST_VERSION,
        statusCallback=None,
        cancelRequested=None,
    ):
        if not inputVolumeNode:
            raise ValueError("No input volume node provided.")

        self.ensureDependencies()

        tempDir = tempfile.mkdtemp(prefix="MuscleMap_")
        inputPath = os.path.join(tempDir, "input.nii.gz")
        outputDir = os.path.join(tempDir, "output")

        logging.info(f"[MuscleMap] Saving input volume to: {inputPath}")
        if not slicer.util.saveNode(inputVolumeNode, inputPath):
            raise RuntimeError(f"Failed to save input volume to {inputPath}")

        # Run mm_segment with Slicer's own interpreter instead of relying on the
        # `mm_segment` entry point being on PATH: this guarantees the package we
        # installed (and read the label names from) is the one that runs.
        s_value = "75" if force_lower_overlap else "90"
        cmd = [sys.executable, "-m", "scripts.mm_segment", "-i", inputPath, "-o", outputDir, "-s", s_value]
        cmd.extend(["-g", "N" if force_cpu else "Y"])
        cmd.extend(["-r", region])
        if model_version and model_version != LATEST_VERSION:
            cmd.extend(["--model_version", model_version])

        logging.info("[MuscleMap] Running command: " + " ".join(cmd))

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
            logging.info("[MuscleMap] Forcing CPU execution (CUDA_VISIBLE_DEVICES='').")

        try:
            output = self._runProcess(cmd, tempDir, env, statusCallback, cancelRequested)
        except SegmentationCancelled:
            shutil.rmtree(tempDir, ignore_errors=True)
            raise

        # mm_segment writes <input>_dseg.nii.gz into the output directory
        outputs = sorted(glob.glob(os.path.join(outputDir, "*_dseg.nii*")))
        if not outputs:
            for ext in ("*.nii.gz", "*.nii", "*.nrrd"):
                outputs.extend(glob.glob(os.path.join(outputDir, ext)))
        if not outputs:
            # keep tempDir for debugging
            tail = "".join(output.splitlines(keepends=True)[-30:])
            raise RuntimeError(
                "mm_segment did not produce a segmentation.\n\n"
                f"Working directory (kept for inspection): {tempDir}\n\n"
                f"Last output lines:\n{tail}"
            )
        outputPath = outputs[0]
        logging.info(f"[MuscleMap] Using output file: {outputPath}")

        if statusCallback:
            statusCallback("Loading the segmentation into Slicer.")

        labelNode = slicer.util.loadLabelVolume(outputPath)
        if not labelNode:
            raise RuntimeError(f"Failed to load the MuscleMap output labelmap: {outputPath}")

        # --- APPLY a real Slicer ColorTable (names+colors) ---
        labels = []
        try:
            colorNode, labels = self._build_colortable_from_labels_json(region, model_version, output)
        except Exception as e:
            logging.warning(f"[MuscleMap] Could not build the label color table: {e}")
            slicer.util.warningDisplay(
                "The segmentation finished, but the muscle label names could not be loaded.\n"
                "Segments will be named by their label value instead.\n\n"
                f"Details: {e}",
                windowTitle="MuscleMap – label names unavailable",
            )
            colorNode = None

        # The import below takes the segment names from the labelmap's color table
        if colorNode is not None:
            labelDisplayNode = labelNode.GetDisplayNode()
            if labelDisplayNode:
                labelDisplayNode.SetAndObserveColorNodeID(colorNode.GetID())
            logging.info("[MuscleMap] Applied ColorTable to labelmap display.")

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", f"MuscleMap_{region}_Segmentation"
        )
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentationNode)

        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.GetDisplayNode().SetVisibility(True)
        # Export uses the input image geometry
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolumeNode)

        if colorNode is not None:
            # Remember the color table so that exports keep the original
            # MuscleMap label values instead of renumbering segments 1..N.
            segmentationNode.SetNodeReferenceID(self.COLOR_TABLE_REFERENCE_ROLE, colorNode.GetID())
            if hasattr(segmentationNode, "SetLabelmapConversionColorTableNodeID"):
                segmentationNode.SetLabelmapConversionColorTableNodeID(colorNode.GetID())
            segmentationNode.SetAttribute(self.LABELS_ATTRIBUTE, json.dumps(labels))

        slicer.mrmlScene.RemoveNode(labelNode)
        slicer.util.setSliceViewerLayers(background=inputVolumeNode)

        shutil.rmtree(tempDir, ignore_errors=True)
        return segmentationNode

    def defaultExportPath(self, segmentationNode) -> str:
        """Suggest <input image folder>/<input image name>_dseg.nii.gz (MuscleMap convention)."""
        referenceVolume = segmentationNode.GetNodeReference(
            slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()
        )
        directory = ""
        baseName = segmentationNode.GetName().replace("_Segmentation", "")
        if referenceVolume:
            baseName = referenceVolume.GetName()
            storageNode = referenceVolume.GetStorageNode()
            inputFile = storageNode.GetFileName() if storageNode else None
            if inputFile:
                directory = os.path.dirname(inputFile)
                baseName = os.path.basename(inputFile)
                for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
                    if baseName.lower().endswith(ext):
                        baseName = baseName[: -len(ext)]
                        break
        if not directory or not os.path.isdir(directory):
            directory = slicer.app.defaultScenePath
        return os.path.join(directory, f"{baseName}_dseg.nii.gz")

    def exportSegmentation(self, segmentationNode, labelmapPath: str):
        """Write a NIfTI labelmap (MuscleMap label values), a CSV label table and a color table.

        Returns the list of written file paths.
        """
        # Always write <name>_dseg.nii.gz, the MuscleMap naming convention
        base = labelmapPath
        for ext in (".nii.gz", ".nii"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
                break
        if not base.lower().endswith("_dseg"):
            base += "_dseg"
        labelmapPath = base + ".nii.gz"

        colorNode = segmentationNode.GetNodeReference(self.COLOR_TABLE_REFERENCE_ROLE)
        referenceVolume = segmentationNode.GetNodeReference(
            slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()
        )
        extentMode = (
            slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
            if referenceVolume
            else slicer.vtkSegmentation.EXTENT_UNION_OF_EFFECTIVE_SEGMENTS
        )

        segmentIds = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(segmentIds)

        segmentationsLogic = slicer.modules.segmentations.logic()
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "MuscleMapExport")
        written = []
        try:
            exportedWithColorTable = False
            if colorNode is not None:
                try:
                    ok = segmentationsLogic.ExportSegmentsToLabelmapNode(
                        segmentationNode, segmentIds, labelmapNode, referenceVolume, extentMode, colorNode
                    )
                    exportedWithColorTable = True
                except TypeError:
                    # Older Slicer without the colorTableNode argument: remap below
                    ok = segmentationsLogic.ExportSegmentsToLabelmapNode(
                        segmentationNode, segmentIds, labelmapNode, referenceVolume, extentMode
                    )
            else:
                ok = segmentationsLogic.ExportSegmentsToLabelmapNode(
                    segmentationNode, segmentIds, labelmapNode, referenceVolume, extentMode
                )
            if not ok:
                raise RuntimeError("Slicer could not convert the segmentation to a labelmap.")

            if colorNode is not None and not exportedWithColorTable:
                self._remapToColorTableValues(segmentationNode, segmentIds, labelmapNode, colorNode)

            if not slicer.util.saveNode(labelmapNode, labelmapPath):
                raise RuntimeError(f"Failed to write {labelmapPath}")
            written.append(labelmapPath)
        finally:
            slicer.mrmlScene.RemoveNode(labelmapNode)

        csvPath = base + "_labels.csv"
        labelsJson = segmentationNode.GetAttribute(self.LABELS_ATTRIBUTE)
        if labelsJson:
            labels = sorted(json.loads(labelsJson), key=lambda x: int(x["value"]))
            with open(csvPath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["value", "region", "anatomy", "side"])
                for item in labels:
                    writer.writerow([item["value"], item.get("region", ""), item.get("anatomy", ""), item.get("side", "")])
            written.append(csvPath)
        elif colorNode is not None:
            with open(csvPath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["value", "label"])
                for value in range(colorNode.GetNumberOfColors()):
                    name = colorNode.GetColorName(value)
                    if name and name != "background":
                        writer.writerow([value, name])
            written.append(csvPath)

        if colorNode is not None:
            ctblPath = base + ".ctbl"
            if slicer.util.saveNode(colorNode, ctblPath):
                written.append(ctblPath)
            else:
                logging.warning(f"[MuscleMap] Could not write color table to {ctblPath}")

        logging.info("[MuscleMap] Exported: " + ", ".join(written))
        return written

    def _remapToColorTableValues(self, segmentationNode, segmentIds, labelmapNode, colorNode):
        """Replace the consecutive export values (1..N) by the color table values of the segment names."""
        import numpy as np

        segmentation = segmentationNode.GetSegmentation()
        exported = slicer.util.arrayFromVolume(labelmapNode)
        remapped = np.zeros_like(exported)
        for i in range(segmentIds.GetNumberOfValues()):
            segment = segmentation.GetSegment(segmentIds.GetValue(i))
            value = colorNode.GetColorIndexByName(segment.GetName())
            if value < 0:
                logging.warning(f"[MuscleMap] Segment '{segment.GetName()}' not in color table, label value kept as {i + 1}.")
                value = i + 1
            remapped[exported == (i + 1)] = value
        slicer.util.updateVolumeFromArray(labelmapNode, remapped)
