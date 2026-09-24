import numpy as np
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QDialog, QProgressBar, QHBoxLayout
from qfluentwidgets import PushButton, Slider, ComboBox, SpinBox, LineEdit, MessageBox, Dialog, ProgressBar, BodyLabel, DoubleSpinBox, SwitchButton
from PySide6.QtCore import Qt, QEventLoop, QSignalBlocker, Signal
import trimesh
import os, json
import natsort
import traceback






class SafeGenerator:
    def __init__(self, morph_num, root_dir:str=''):
        self.base = trimesh.load(os.path.join(root_dir, 'safe.obj'), process=False)

        self.base_vertices = np.array(self.base.vertices)
        self.base_faces = np.array(self.base.faces)
        self.morph_num = morph_num
        self.all_morphs = []
        for i in range(morph_num):
            self.all_morphs.append(self.get_blend_shape(os.path.join(root_dir, f'safe_morph_{i+1:02d}.obj'), self.base_vertices))
            
        self.weightFunc = lambda x: x * 1.5 - 0.5
          
    @staticmethod
    def get_blend_shape(path, base):
        morph = trimesh.load(path, process=False)
        morph_vertices = np.array(morph.vertices)
        offset = morph_vertices - base
        return offset
            
            
    def weightFunc(self, func:callable,):
        self.weightFunc = func
        
        
    def _getJoint(self, vertices):
        ref_vertex = vertices[[6646, 7140]]
        ref_vertex2 = vertices[10501]
        joint_loc = np.mean(ref_vertex, axis=0)
        joint_loc[2] = 0.0
        joint_loc[0] = ref_vertex2[0]
        joint_axis = np.array([0.0, 0.0, 1.0])
        
        return joint_loc, joint_axis
        
    def generate(self, morph_values):
        assert len(morph_values) == len(self.all_morphs), "Morph values length must match number of morphs."
        
        blended_vertices = np.array(self.base_vertices)
        for i, weight in enumerate(morph_values):
            weight = self.weightFunc(weight)
            if weight < 0:
                weight = 0.0
            blended_vertices += self.all_morphs[i] * weight

        joint_loc, joint_axis = self._getJoint(blended_vertices)

        return blended_vertices, self.base_faces, joint_loc, joint_axis

class Script(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Blend Shape Viewer and Corrector")
        layout = QVBoxLayout(self)
        
        self.base = trimesh.load('safe.obj', process=False)

        self.base_vertices = np.array(self.base.vertices)
        self.base_faces = np.array(self.base.faces)
        
        self.generator = SafeGenerator(20)
        
        
        self.all_morphs_sliders = []
        for i in range(self.generator.morph_num):
            _hlayout = QHBoxLayout()
            morph_slider = Slider(Qt.Horizontal)
            morph_slider.setRange(0, 100)
            morph_slider.setValue(0)
            morph_slider.setToolTip(f'Morph {i+1}')
            
            _hlayout.addWidget(BodyLabel(f'Morph {i+1:02d}'))
            _hlayout.addWidget(morph_slider)
            layout.addLayout(_hlayout)
            morph_slider.valueChanged.connect(self.update_view)
            self.all_morphs_sliders.append(morph_slider)

        randomize_button = PushButton("Randomize Morphs")
        randomize_button.clicked.connect(self.randomize_morphs)
        layout.addWidget(randomize_button)

        self.setLayout(layout)
        
        self.resize(500, 300)
        
        self.trigger_auto = True
        
    
    
    def show_mesh(self, blend_mesh, joint):

        b3d.clear()
        b3d.add({'mesh': blend_mesh, 'joint_line': joint})
        
        
    def update_view(self):
        if self.trigger_auto:
            weight = [slider.value() / 100.0 for slider in self.all_morphs_sliders]
            blended_vertices, faces, joint_loc, joint_axis = self.generator.generate(weight)
            joint = np.stack([joint_loc, joint_loc + joint_axis], axis=0)
            blend_mesh = {'vertex': blended_vertices, 'face': faces}
            self.show_mesh(blend_mesh, joint)
            
    def randomize_morphs(self):
        self.trigger_auto = False
        for slider in self.all_morphs_sliders:
            slider.setValue(np.random.randint(0, 101))
        self.trigger_auto = True
        self.update_view()



if __name__ == "__main__":
    app = QApplication([])
    script = Script()
    script.show()
    app.exec()
    
else:
    script = Script()
    script.show()

