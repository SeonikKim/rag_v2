import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  uploadPdf(form: FormData): Observable<any> {
    return this.http.post('/upload', form);
  }

  query(question: string): Observable<any> {
    return this.http.post('/query', { question });
  }
}
