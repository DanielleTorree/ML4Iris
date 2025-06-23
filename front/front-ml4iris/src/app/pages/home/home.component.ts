import { Component } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';

import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatDividerModule } from '@angular/material/divider';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { Iris } from '../../shared/interfaces/Iris';
import { IrisService } from '../../shared/services/iris.service';
import { IrisInput } from '../../shared/interfaces/IrisInput';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    FormsModule, 
    MatFormFieldModule, 
    MatInputModule,
    CommonModule,
    MatButtonModule, 
    MatDividerModule, 
    MatIconModule,
    MatTableModule
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {
  form: FormGroup;

  displayedColumns: string[] = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'specie'];
  irisList: Iris[] = [];

  constructor(
    private fb: FormBuilder, 
    private irisService: IrisService
  ) {
    this.form = this.fb.group({
      sepalLength: ['', [Validators.required, Validators.min(0), Validators.max(100)]], 
      sepalWidth: ['', [Validators.required, Validators.min(0), Validators.max(100)]],
      petalLength: ['', [Validators.required, Validators.min(0), Validators.max(100)]],
      petalWidth: ['', [Validators.required, Validators.min(0), Validators.max(100)]],
    });
  }

  ngOnInit() {
    this.getIrisList();
  }

  onSubmit() {
    if (this.form.valid) {
      console.log('Formulário válido:', this.form.value);
      
      const iris: IrisInput = {
        sepal_length_cm: this.form.get('sepalLength')?.value,
        sepal_width_cm: this.form.get('sepalWidth')?.value,
        petal_length_cm: this.form.get('petalLength')?.value,
        petal_width_cm: this.form.get('petalWidth')?.value,
      };

      this.saveIris(iris);

    } else {
      console.log('Formulário inválido');
    }
  }

  getIrisList() {
    this.irisService.getIrisList()
      .subscribe({
        next: data => {
          this.irisList = data;
        },
        error: err => {
          console.error("Erro ao carregar dados: ", err)
        }
      })
  }
  
  saveIris(data: IrisInput) {
    console.log("Corpo enviado:", data);

    this.irisService.saveIris(data)
      .subscribe({
        next: data => {
          alert("Iris salva com sucesso!")
          
          
          this.form.reset();
          this.irisList = [];
          this.getIrisList();
        },
        error: err => {
          alert("Erro ao tentar salvar iris!")
          console.error("Erro ao carregar dados: ", err)
        }
      })
  }
}
