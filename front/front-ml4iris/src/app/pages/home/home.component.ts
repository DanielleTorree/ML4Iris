import { Component } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';

import { Iris } from '../../shared/interfaces/Iris';
import { IrisService } from '../../shared/services/iris.service';
import { IrisInput } from '../../shared/interfaces/IrisInput';

import { MessageService } from '../../shared/services/message.service';

import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatDividerModule } from '@angular/material/divider';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { MatSnackBarModule } from '@angular/material/snack-bar';

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
    MatTableModule,
    MatSnackBarModule
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {
  form: FormGroup;

  displayedColumns: string[] = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'specie', 'actions'];
  irisList: Iris[] = [];

  constructor(
    private fb: FormBuilder, 
    private irisService: IrisService,
    private messageService: MessageService
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
      const iris: IrisInput = {
        sepal_length_cm: this.form.get('sepalLength')?.value,
        sepal_width_cm: this.form.get('sepalWidth')?.value,
        petal_length_cm: this.form.get('petalLength')?.value,
        petal_width_cm: this.form.get('petalWidth')?.value,
      };

      this.saveIris(iris);
    } 
  }

  getIrisList() {    
    this.irisService.getIrisList()
      .subscribe({
        next: data => {
          this.irisList = data;
        },
        error: err => {
          this.messageService.error("Erro ao tentar carregar dados da tabela de iris!");
        }
      })
  }
  
  saveIris(data: IrisInput) {
    this.irisService.saveIris(data)
      .subscribe({
        next: data => {
          this.messageService.success("Iris salva com sucesso!")
          
          this.form.reset();
          this.irisList = [];
          this.getIrisList();
        },
        error: err => {
          this.messageService.error("Erro ao tentar salvar iris!");
        }
      })
  }

  deleteIris(element: Iris) {
    this.irisService.deleteIris(element.id)
      .subscribe({
        next: data => {
          this.messageService.success("Iris excluída com sucesso!")
          
          this.form.reset();
          this.irisList = [];
          this.getIrisList();
        },
        error: err => {
          this.messageService.error("Erro ao tentar excluir iris!");
        }
      })
  }

  getSpecieClass(specie: string): string {
    switch (specie.toLowerCase()) {
      case 'iris-setosa':
        return 'setosa';
      case 'iris-versicolor':
        return 'versicolor';
      case 'iris-virginica':
        return 'virginica';
      default:
        return '';
    }
  }
}
