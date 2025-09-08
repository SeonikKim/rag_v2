import { Component } from '@angular/core';
import { ApiService } from './api.service';

@Component({
  selector: 'app-root',
  template: `
    <input type="file" (change)="onFile($event)" />
    <input [(ngModel)]="question" placeholder="Ask" />
    <button (click)="ask()">Ask</button>
    <pre>{{ answer }}</pre>
  `,
})
export class AppComponent {
  question = '';
  answer = '';

  constructor(private api: ApiService) {}

  onFile(evt: any) {
    const file = evt.target.files[0];
    const form = new FormData();
    form.append('file', file);
    this.api.uploadPdf(form).subscribe();
  }

  ask() {
    this.api.query(this.question).subscribe((res) => (this.answer = res.answer));
  }
}
